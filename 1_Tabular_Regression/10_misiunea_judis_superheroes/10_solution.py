import pandas as pd
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

# ---- Task 1 ----
Task1 = len(dfTrain["Publisher"].unique())

# ---- Task 2 ----
ramase = dfTrain[dfTrain["Alignment"] == "good"]
Task2 = ramase["Publisher"].mode()[0]

# ---- Task 3 & 4 ----
X_train = dfTrain.drop(["id", "powerstats__combat", "Super Strength"], axis=1)
y_train_R = dfTrain["powerstats__combat"]
y_train_C = dfTrain["Super Strength"]

X_test = dfTest.drop(["id"], axis=1)

cat_cols = dfTrain.select_dtypes(include=["object", "string"]).columns.tolist()

X_train[cat_cols] = X_train[cat_cols].astype(str)
X_test[cat_cols] = X_test[cat_cols].astype(str)

modelR = CatBoostRegressor(
    learning_rate=0.1,
    iterations=1000,
    random_seed=42,
    cat_features = cat_cols
)

modelR.fit(X_train, y_train_R)
Task3 = modelR.predict(X_test)

modelC = CatBoostClassifier(
    learning_rate=0.1,
    iterations=1000,
    random_seed=42,
    cat_features = cat_cols
)

modelC.fit(X_train, y_train_C)
Task4 = modelC.predict(X_test)

# ---- Submission ----
rows = []

rows.append({
    "id": "GLOBAL",
    "subtaskID": "task1",
    "answer": Task1
})

rows.append({
    "id": "GLOBAL",
    "subtaskID": "task2",
    "answer": Task2
})

for pID, t3 in zip(dfTest["id"], Task3):
    rows.append({
        "id": pID,
        "subtaskID": "task3",
        "answer": round(t3, 2)
    })
    
for pID, t4 in zip(dfTest["id"], Task4):
    rows.append({
        "id": pID,
        "subtaskID": "task4",
        "answer": t4
    })

submission = pd.DataFrame(rows, columns=["id", "subtaskID", "answer"])
submission.to_csv("submission.csv", index = False)