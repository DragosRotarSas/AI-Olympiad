import pandas as pd
import numpy as np
import re

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words("english"))

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

# ======================
# REZOLVARE TASK 1 2, 3
# ======================

# ---- Task 1 ----
medie_len = (dfTest["sentence1"].str.len() + dfTest["sentence2"].str.len()) / 2
conditii_s1 = [
    medie_len < 50,
    (medie_len >= 50) & (medie_len < 100),
    medie_len >= 100
]
optiuni_s1 = ["Short", "Medium", "Long"]
ans_subtask_1 = np.select(conditii_s1, optiuni_s1, default="")

# ---- Task 2 ----
nr_cuvinte_s1 = dfTest["sentence1"].str.split().str.len()
nr_cuvinte_s2 = dfTest["sentence2"].str.split().str.len()
ans_subtask_2 = nr_cuvinte_s1 + nr_cuvinte_s2

# ---- Task 3 ----
ans_subtask_3 = abs(dfTest["sentence1"].str.len() - dfTest["sentence2"].str.len())

# ====================
# REZOLVARE SUBTASK 4
# ====================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

for df in [dfTrain, dfTest]:
    df["s1_clean"] = df["sentence1"].apply(clean_text)
    df["s2_clean"] = df["sentence2"].apply(clean_text)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", sublinear_tf=True)
toate_textele = pd.concat([
    dfTrain["s1_clean"], dfTrain["s2_clean"], dfTest["s1_clean"], dfTest["s2_clean"]
])
vectorizer.fit(toate_textele)

def extract_features(df):
    tfidf_s1 = vectorizer.transform(df["s1_clean"])
    tfidf_s2 = vectorizer.transform(df["s2_clean"])

    matrice_sim = cosine_similarity(tfidf_s1, tfidf_s2)
    cosine_sim = np.diag(matrice_sim)

    features = pd.DataFrame({
        "cosine_sim": np.asarray(cosine_sim).flatten(),
        "len_diff": abs(df["sentence1"].str.len() - df["sentence2"].str.len()),
        "word_diff": abs(df["sentence1"].str.split().str.len() - df["sentence2"].str.split().str.len())
    })
    return features


X_train_features = extract_features(dfTrain)
y_train = dfTrain["score"]

X_test_features = extract_features(dfTest)

model = HistGradientBoostingRegressor(
    random_state=42)
model.fit(X_train_features, y_train)
ans_subtask_4 = model.predict(X_test_features)

ans_subtask_4 = np.clip(ans_subtask_4, 0.0, 5.0)

# ===========
# SUBMISSION
# ===========

answers = []
for dID, a1, a2, a3, a4 in zip(dfTest["sampleID"], ans_subtask_1, ans_subtask_2, ans_subtask_3, ans_subtask_4):
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