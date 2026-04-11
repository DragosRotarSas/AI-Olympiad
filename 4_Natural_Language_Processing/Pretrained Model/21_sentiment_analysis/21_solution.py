import re
import tqdm

import pandas as pd
import numpy as np
import torch

from transformers import BertModel, BertTokenizer
from catboost import CatBoostClassifier, Pool

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ====
# BERT
# ====

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
nume_model = "bhadresh-savani/bert-base-uncased-emotion"

tokenizer = BertTokenizer.from_pretrained(nume_model)

bert = BertModel.from_pretrained(
    nume_model,
    output_hidden_states=True,
    return_dict=True,
    ignore_mismatched_sizes=True
)
bert = bert.to(device)

bert.eval()

# ====
# DATA
# ====

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

def getData(df, batch_size=32):
    X = []
    texte_curate = []

    for text in df["text"]:
        text = str(text)
        text = re.sub("im", "i am", text)
        text = re.sub("ive", "i have", text)
        texte_curate.append(text)

    with torch.inference_mode():

        for i in tqdm.tqdm(range(0, len(texte_curate), batch_size), desc="Procesez in batch-uri"):

            batch_texte = texte_curate[i: i + batch_size]

            encoded = tokenizer(
                batch_texte,
                max_length=128,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            segments = encoded["token_type_ids"].to(device)

            output = bert(input_ids, attention_mask=mask, token_type_ids=segments)

            last_hidden = output["last_hidden_state"]

            # --- MEAN POOLING VECTORIZAT ---

            # Aliniem masca 2D la dimensiunea 3D a tensorului pentru a le putea inmulti
            mask_expanded = mask.unsqueeze(-1).expand(last_hidden.size()).float()

            # Anulam ce e extra din padding prin inmultire si adunam tokenii valizi intr-un singur rand
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)

            # Numaram tokenii reali din propozitie (evitam impartire la 0)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

            # Impartim suma totala la numarul de tokeni pentru a calcula media (Mean Pooling)
            sentence_vectors = sum_embeddings / sum_mask

            X.extend(sentence_vectors.cpu().numpy().tolist())

    return X

X_train = getData(dfTrain)
y_train = dfTrain["label"]
X_test = getData(dfTest)

# ==========
# SPLIT TEST
# ==========

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size = 0.2
)

train_pool = Pool(
    data=X_tr,
    label=y_tr
)

val_pool = Pool(
    data=X_val
)

test_pool = Pool(
    data=X_test
)

# ========
# CATBOOST
# ========

cat = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.1,
    loss_function="MultiClass",
    random_state=42,
    task_type="GPU",
    verbose=500,
)

cat.fit(train_pool)

val_pred = cat.predict(val_pool)
scor_local = f1_score(y_val, val_pred, average="macro")
print(f"Scorul F1 pe setul de validare este: {scor_local}")

pred_test = cat.predict(test_pool)

# ==========
# SUBMISSION
# ==========

answers = []

for i, p in zip(dfTest["SampleID"], pred_test):
    answers.append({
        "SampleID": i,
        "label": p[0]
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index=False)