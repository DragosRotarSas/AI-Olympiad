import pandas as pd
import numpy as np

import re
import nltk
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

#nltk.download("punkt_tab")
#nltk.download("stopwords")

dfTrain = pd.read_csv("train_data.csv")
dfTest = pd.read_csv("test_data.csv")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def getCleanTokens(text):

    text = text.lower()


    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)

    #text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"http\S+|www\S+", "", text)  # links


    #text = re.sub(r'(.)\1{2,}', r'\1', text)  # looool -> lol
    #text = re.sub(r'(.+?)\1{2,}', r'\1', text)  # abcabcabcabc -> abc

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r'\d+', '', text)

    tokens = text.split()

    tokens_clean = [
        lemmatizer.lemmatize(stemmer.stem(word))
        for word in tokens if word not in stop_words
    ]

    return " ".join(tokens_clean)

def getData(df):
    X = []
    for _, row in df.iterrows():
        X.append(getCleanTokens(row["comment_text"]))
    df["clean text"] = X

getData(dfTrain)
getData(dfTest)

vectortizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    norm="l2"
)

X_train = vectortizer.fit_transform(dfTrain["clean text"])
X_test = vectortizer.transform(dfTest["clean text"])

# toxic
y_t = dfTrain["toxic"]
model_t = LinearSVC(random_state=42, class_weight='balanced')
model_t.fit(X_train, y_t)
pred_t = model_t.predict(X_test)

# severe_toxic
y_st = dfTrain["severe_toxic"]
model_st = LinearSVC(random_state=42, class_weight='balanced')
model_st.fit(X_train, y_st)
pred_st = model_st.predict(X_test)

# obscene
y_o = dfTrain["obscene"]
model_o = LinearSVC(random_state=42, class_weight='balanced')
model_o.fit(X_train, y_o)
pred_o = model_o.predict(X_test)

# insult
y_i = dfTrain["insult"]
model_i = LinearSVC(random_state=42, class_weight='balanced')
model_i.fit(X_train, y_i)
pred_i = model_i.predict(X_test)

# ==========
# Submission
# ==========

answers = []

for dID, t, st, o, i in zip(dfTest["id"], pred_t, pred_st, pred_o, pred_i):
    answers.append({
        "subtaskID": 1,
        "datapointID": dID,
        "answer": np.array([t, st, o, i]).tolist()
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index=False)