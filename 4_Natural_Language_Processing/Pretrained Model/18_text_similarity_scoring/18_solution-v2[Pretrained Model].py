import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

from tqdm import tqdm

#!pip install catboost
from catboost import CatBoostRegressor, Pool

# ========
# DATASETS
# ========

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# REZOLVARE TASK 1 - 3
# ====================

# ---- Task 1 ----
medie_len = (dfTest["sentence1"].str.len() + dfTest["sentence2"].str.len()) / 2
conditii_s1 = [
    medie_len < 50,
    (medie_len >= 50) & (medie_len < 100),
    medie_len >= 100
]
optiuni_s1 = ["Short", "Medium", "Long"]
Task1 = np.select(conditii_s1, optiuni_s1, default="")

# ---- Task 2 ----
nr_cuvinte_s1 = dfTest["sentence1"].str.split().str.len()
nr_cuvinte_s2 = dfTest["sentence2"].str.split().str.len()
Task2 = nr_cuvinte_s1 + nr_cuvinte_s2

# ---- Task 3 ----
Task3 = abs(dfTest["sentence1"].str.len() - dfTest["sentence2"].str.len())

# ================
# REZOLVARE TASK 4
# ================

# ==========
# GET TOKENS
# ==========

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def getTokens(sentence1, sentence2):
    # Aceasta singura linie face tot: adauga [CLS], [SEP] la mijloc, [SEP] la final,
    # calculeaza segment_ids corect (0 si 1) si taie in siguranta la maxim 512
    encoded = tokenizer(
        sentence1,
        sentence2,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device), encoded["token_type_ids"].to(device)

# =====
# MODEL
# =====

# Acum incarcam modelul
model = BertModel.from_pretrained(
    'bert-base-uncased',
    output_hidden_states = True,    # Daca e False (default), iti da rezultatul final, cand e True iti da rezultatul de la mai multe layere, ce de multe ori are o semnificatie semantica mai buna
    return_dict = True  # Returneaza sub forma de dictionar, cand e False e sub forma de Tuple si deci trebuie accesate prin [0], [1] etc.
)

model = model.to(device)

# ==================
# PREPROCESSING DATA
# ==================

# Si il punem in modul de evaluare
model.eval()

def getData(df):
    X = []
    with torch.inference_mode():

        for i, row in tqdm(df.iterrows(), total = len(df), desc = "Procesez propozitii"):

            # Obtinem vectorul de la BERT
            tokens, mask, segments = getTokens(row["sentence1"], row["sentence2"])
            output = model(tokens, mask, segments)

            # Luam outputul si il punem ca lista
            bert_features = output["pooler_output"].cpu().numpy()[0].tolist()

            # Adaugam feature-uri custom
            s1 = row["sentence1"]
            s2 = row["sentence2"]

            len_diff = abs(len(s1) - len(s2))
            word_diff = abs(len(s1.split()) - len(s2.split()))

            set1 = set(s1.lower().split())
            set2 = set(s2.lower().split())
            common_words = len(set1.intersection(set2))

            bert_features.extend([len_diff, word_diff, common_words])

            X.append(bert_features)
    return X

X_train = getData(dfTrain)
X_test = getData(dfTest)

X_train = np.array(X_train)
y_train = dfTrain["score"]
X_test = np.array(X_test)

# ===============
# K-FOLD CATBOOST
# ===============

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

# Adunarea predictiilor modelului
test_predictions = np.zeros(len(X_test))

# Loop-ul principal
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n--- Antrenez Fold {fold + 1} / {K} ---")

    # 1. Impartim datele folosind index-urile generate de KFold
    X_tr = X_train[train_idx]
    y_tr = y_train.iloc[train_idx]  # Folosim .iloc pentru că y_train e Pandas Series

    X_val = X_train[val_idx]
    y_val = y_train.iloc[val_idx]

    train_pool = Pool(data=X_tr, label=y_tr)
    val_pool = Pool(data=X_val, label=y_val)

    cat_fold = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.1,
        loss_function="MAE",
        random_state=42,
        task_type="GPU",
        verbose=500,
        random_strength=1,
        depth=4,
        l2_leaf_reg=3
    )

    cat_fold.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100
    )

    test_predictions += cat_fold.predict(X_test)

Task4 = test_predictions / K
Task4 = np.clip(Task4, 0.0, 5.0)

"""
param_grid = {
    'depth': [4, 6, 8],                  
    'learning_rate': [0.01, 0.03, 0.1], 
    'l2_leaf_reg': [1, 3, 5, 9],         
    'random_strength': [1, 5]           
}

rezultate = cat.randomized_search(
    param_distributions = param_grid,
    X=train_pool,
    cv=3,
    n_iter=10,
    verbose=1000
)

# OUTPUT: {'random_strength': 1, 'depth': 4, 'learning_rate': 0.1, 'l2_leaf_reg': 3}


Task4 = cat.predict(test_pool)
Task4 = np.clip(Task4, 0.0, 5.0)

print("Cei mai buni parametrii:")
print(rezultate["params"])

"""

# ==========
# SUBMISSION
# ==========

answers = []

for dID, a1, a2, a3, a4 in zip(dfTest["sampleID"], Task1, Task2, Task3, Task4):
    answers.append({
        "subtaskID": 1,
        "datapointID":dID,
        "answer": a1
    })
    answers.append({
        "subtaskID": 2,
        "datapointID": dID,
        "answer": a2
    })
    answers.append({
        "subtaskID": 3,
        "datapointID": dID,
        "answer": a3
    })
    answers.append({
        "subtaskID": 4,
        "datapointID": dID,
        "answer": round(a4, 4)
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index=False)