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
values = dfTrain["Age"].value_counts()

# ---- Task 2 ----
procentaj = {}
for age in dfTrain["Age"].unique():
    procentaj.update(
        {int(age): float(dfTrain[dfTrain["Age"] == age]["Smoking"].mean() * 100)}
    )

# ---- Task 3 ----
X_train = dfTrain.drop(["PatientID", "Diagnosis"], axis=1)
y_train = dfTrain["Diagnosis"]
X_test = dfTest.drop(["PatientID"], axis=1)

num_cols = X_train.select_dtypes(include=["number"]).columns
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
    ("model", model)
])

c.fit(X_tr, y_tr)
pred = c.predict(X_test)

# ---- Submission ----
rows = []

for p, age, pr in zip(dfTest["PatientID"], dfTest["Age"], pred):
    rows.append({
        "subtaskID": 1,
        "datapointID": p,
        "answer": values[age]
    })
    rows.append({
        "subtaskID": 2,
        "datapointID": p,
        "answer": round(procentaj[age], 2)
    })
    rows.append({
        "subtaskID": 3,
        "datapointID": p,
        "answer": pr
    })

submission = pd.DataFrame(rows, columns=["subtaskID", "datapointID", "answer"])
submission.to_csv("submission.csv", index=False)