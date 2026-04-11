import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

dfTest = pd.read_csv("test.csv")
dfTrain = pd.read_csv("train.csv")

# ---- Task 1 ----
Task1 = (dfTest["Monthly Charge"] > 70).astype(int) + (dfTest["Total Extra Data Charges"] > 10).astype(int)

# ---- Task 2 ----
Task2 = (dfTest["Avg Speed"] < 50).astype(int) + (dfTest["Ping Score"] > 80).astype(int) + (dfTest["Link Quality Index"] < 30).astype(int)

# ---- Task 3 ----
X_train = dfTrain.drop(["SampleID", "City", "Country", "Customer ID", "Lat Long", "Latitude", "Longitude", "Quarter", "State", "Zip Code", "Churn"], axis=1) # Payment Method maybe
y_train = dfTrain["Churn"]
X_test = dfTest.drop(["SampleID", "City", "Country", "Customer ID", "Lat Long", "Latitude", "Longitude", "Quarter", "State", "Zip Code"], axis=1)

num_cols = X_train.select_dtypes(include="number").columns
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns

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
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

c = Pipeline(steps=[
    ("prep", preprocess),
    ("mod", model)
])

c.fit(X_tr, y_tr)
pred = c.predict(X_test)

# ---- Submission ----
rows = []

for sID, ans in zip(dfTest["SampleID"], Task1):
    rows.append({
        "id": sID,
        "subtaskID": 1,
       "answer": ans
    })

for sID, ans in zip(dfTest["SampleID"], Task2):
    rows.append({
        "id": sID,
        "subtaskID": 2,
       "answer": ans
    })

for sID, ans in zip(dfTest["SampleID"], pred):
    rows.append({
        "id": sID,
        "subtaskID": 3,
       "answer": ans
    })

submission = pd.DataFrame(rows, columns=["id", "subtaskID", "answer"])
submission.to_csv("submission.csv", index=False)