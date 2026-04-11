import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

#nltk.download("punkt_tab")
#nltk.download("stopwords")

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

def getTokens(text):
    text = text.lower()
    #text = text.replace('ş', 'ș').replace('ţ', 'ț')
    return text

dfTrain["clean text"] = dfTrain["Versuri"].apply(getTokens)
dfTest["clean text"] = dfTest["Versuri"].apply(getTokens)

# ===================
# 1. FEATURE UNION
# ===================

# Rime si Stil
char_vectorizer = TfidfVectorizer(
    ngram_range=(2, 6),
    norm="l2",
    sublinear_tf=True,  # 1 + log(frecventa) pentru a atribui ok importanta
    analyzer="char_wb"
)

# Cuvinte
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    norm="l2",
    sublinear_tf=True,
    analyzer="word"
)

# Mega-matrice
combiner = FeatureUnion([
    ("char", char_vectorizer),
    ("word", word_vectorizer)
])


X_train = combiner.fit_transform(dfTrain["clean text"])
X_test = combiner.transform(dfTest["clean text"])
y_train = dfTrain["Autor"]

# ==================
# 2. TUNING LA SVC
# ==================

model = LinearSVC(
    random_state=42,
    penalty="l2",
    multi_class="ovr",
    max_iter=5000,
    class_weight="balanced"
)

param_grid = {
    "C": [0.1, 0.5, 1.0, 2.0]   # Regularizare
}

tuner = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

tuner.fit(X_train, y_train)
best_model = tuner.best_estimator_

print(f"Cel mai bun parametru C gasit: {tuner.best_params_['C']}")

pred = best_model.predict(X_test)

# ============
# SUBMISSION
# ============

answers = []

for iD, p in zip(dfTest["Id"], pred):
    answers.append({
        "Id": iD,
        "Autor": p
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index=False)

"""

# =============
# Visualization
# =============

feature_names = tfidf_vectorizer.get_feature_names_out()
autori = model.classes_

for autor in range(0, 7):
    coeficienti = model.coef_[autor]
    top_indici = np.argsort(coeficienti)[-5:]

    top_features = feature_names[top_indici]
    top_coeficienti = coeficienti[top_indici]

    plt.figure(figsize=(8, 8))
    sns.barplot(y=top_features, x=top_coeficienti)
    plt.title(f"{autori[autor]}")
    plt.show()

# ======
# kMeans
# ======


kmeans = KMeans(n_clusters=7, random_state=42, n_init="auto")
train_clusters = kmeans.fit_predict(X_train)

svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X_train)

top_indici_axa_x = svd.components_[0].argsort()[::-1][:10] # Top 10 cele mai importante n-grame pt Axa X
top_indici_axa_y = svd.components_[1].argsort()[::-1][:10] # Top 10 cele mai importante n-grame pt Axa Y

print([feature_names[i] for i in top_indici_axa_x])
print([feature_names[i] for i in top_indici_axa_y])


df_vis = pd.DataFrame({
    'Axa_X': X_2d[:, 0],
    'Axa_Y': X_2d[:, 1],
    'Cluster': train_clusters
})

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_vis,
    x='Axa_X',
    y='Axa_Y',
    hue='Cluster',
    palette='tab10',
    alpha=0.7
)

plt.title("Vizualizarea Clusterelor de Poezii (KMeans + SVD)")
plt.xlabel("Componenta SVD 1 (Stil Principal)")
plt.ylabel("Componenta SVD 2 (Stil Secundar)")
plt.legend(title="Cluster ID")
plt.tight_layout()
plt.show()
"""

