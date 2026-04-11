import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

#----- Task 1 -----
Task1 = int(((dfTest["County"] == "ANDERSON") & (dfTest["School Type"] == "Elementary")).sum())

#----- Task 2 -----
Task2 = dfTest["Academic Growth Rating"].value_counts().idxmax()

#----- Task 3 -----

X_Train = dfTrain.drop(["SampleID", "Relative Performance Rating"], axis=1)
y_Train = dfTrain["Relative Performance Rating"]
X_Test = dfTest.drop(["SampleID"], axis=1)

num_cols = X_Train.select_dtypes(include=["number"]).columns
cat_cols = X_Train.select_dtypes(include=["object", "string"]).columns

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_Train, y_Train,
    test_size=0.2,
    random_state=42,
    stratify=y_Train
)

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

clf = Pipeline(steps=[
    ("prep", preprocess),
    ("mod", model)
])

clf.fit(X_tr, y_tr)
pred = clf.predict(X_Test)

#----- Submission -----
rows = []

rows.append({
    "subtaskID": 1,
    "datapointID": 1,
    "answer": Task1
})

rows.append({
    "subtaskID": 2,
    "datapointID": 2,
    "answer": Task2
})

for dID, a in zip(dfTest["SampleID"], pred):
    rows.append({
        "subtaskID": 3,
        "datapointID": dID,
        "answer": a
    })

submission = pd.DataFrame(rows, columns=["subtaskID", "datapointID", "answer"])
submission.to_csv("submission.csv", index=False)