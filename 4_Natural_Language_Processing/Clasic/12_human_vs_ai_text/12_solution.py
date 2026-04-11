import re

import numpy as np
import pandas as pd
import nltk
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dfTrain = pd.read_csv("train_data.csv")
dfTest = pd.read_csv("test_data.csv")


# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("wordnet")

# ============
# Preprocesare
# ============

def preprocess(text):
    # ---- Transformare text in token-uri ----
    # 1) LOWERCASE
    text = text.lower()

    # 2) REGEX
    # Prima data eliminam tag-uri si link-uri
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)

    # 3) Apoi eliminam ce a mai ramas
    text = re.sub(r"[^\w\săâîșț]", "", text)

    # 4) Despartim in cuvinte
    tokens = nltk.word_tokenize(text=text)

    # 5) Eliminare Stop Words
    stop_words = set(stopwords.words("english"))
    tokens_noStopWords = [word for word in tokens if word not in stop_words]

    # 6) Stemming
    stemmer = PorterStemmer()
    tokens_stemmed = [stemmer.stem(word) for word in tokens_noStopWords]

    # 7) Lemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens_clean = [lemmatizer.lemmatize(word) for word in tokens_stemmed]

    return " ".join(tokens_clean)


def getData(df):
    X = []
    for _, row in df.iterrows():
        X.append(preprocess(row["text"]))
    df["clean text"] = X


getData(dfTrain)
getData(dfTest)

X_train = dfTrain["clean text"]
y_train = dfTrain["label"]
X_test_t1 = dfTest.loc[dfTest["subtaskID"] == 1, "clean text"]
X_test_t2 = dfTest.loc[dfTest["subtaskID"] == 2, "clean text"]

# ======
# Task 1
# ======

# ---- Generare date ----
# 1) Varianta cu BoW
# count_vectorizer = CountVectorizer()
# bow_matrix = count_vectorizer.fit_transform(tokens_clean)

# 2) Varianta cu TFIDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf_vectorizer.fit_transform(X_train)
X_test_t1_vec = tfidf_vectorizer.transform(X_test_t1)

# 3) Varianta cu Cosine Similarity
# similarity_matrix = cosine_similarity(tfidf_matrix)

model_bin = LinearSVC(random_state=42)
model_bin.fit(X_train_vec, y_train)
pred_binar = model_bin.predict(X_test_t1_vec)

# ======
# Task 2
# ======

X_test_t2_vectorized = tfidf_vectorizer.transform(X_test_t2)

model_multi = KMeans(n_clusters=4, n_init=100, random_state=42)
model_multi.fit(X_test_t2_vectorized)
pred_multi = model_multi.labels_

# ---- Vizualizare 1: Bar Charts (inlocuieste WordCloud) ----
terms = tfidf_vectorizer.get_feature_names_out()
centroids = model_multi.cluster_centers_

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for cluster_id in range(4):
    # Luam indicii celor mai puternice 10 cuvinte din centrul clusterului
    top_indices = centroids[cluster_id].argsort()[::-1][:10]
    top_words = [terms[ind] for ind in top_indices]
    top_weights = [centroids[cluster_id][ind] for ind in top_indices]

    axes[cluster_id].barh(top_words[::-1], top_weights[::-1], color='skyblue')
    axes[cluster_id].set_title(f"Cluster {cluster_id}")
    axes[cluster_id].set_xlabel("Scor TF-IDF")

plt.tight_layout()
plt.show()

# ---- Vizualizare 2: PCA Scatter Plot ----
pca = PCA(n_components=2, random_state=42)
# PCA nu accepta sparse matrix, asa ca folosim .toarray()
X_pca = pca.fit_transform(X_test_t2_vectorized.toarray())

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pred_multi, cmap='viridis', alpha=0.6, edgecolors='w')
plt.title("Distributia Clusterelor (PCA 2D)")
plt.xlabel("Componenta Principala 1")
plt.ylabel("Componenta Principala 2")
plt.colorbar(scatter, ticks=[0, 1, 2, 3], label="ID Cluster")
plt.show()

# ==========
# Submission
# ==========

# TODO CRITIC: Cand se afiseaza Bar Chart-urile pe ecran, citeste top cuvinte
# pentru fiecare cluster (0, 1, 2, 3) si corecteaza ordinea in dictionarul asta:
mapare_teme = {
    0: "RELIGION",
    1: "CRIME",
    2: "BUSINESS",
    3: "SCIENCE"
}

answers = []

i = 0
k = 0

for _, row in dfTest.iterrows():
    ID = row["ID"]
    sID = row["subtaskID"]

    if sID == 1:
        answers.append({
            "subtaskID": sID,
            "datapointID": ID,
            "answer": pred_binar[i]
        })
        i = i + 1
    else:
        answers.append({
            "subtaskID": sID,
            "datapointID": ID,
            "answer": mapare_teme[pred_multi[k]]
        })
        k = k + 1

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index=False)