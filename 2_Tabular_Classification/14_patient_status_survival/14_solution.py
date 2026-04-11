import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

q1 = dfTrain["Serum Creatinine"].quantile(0.25)
q2 = dfTrain["Serum Creatinine"].quantile(0.50)
q3 = dfTrain["Serum Creatinine"].quantile(0.75)
mediana_bmi = dfTrain["BMI"].median()

t_stage_counts = dfTrain["T Stage"].value_counts()

def process_tasks(df, is_test=False):

    cond_t1 = [
        (df["GFR"] >= 90),
        (df["GFR"] >= 60) & (df["GFR"] < 90)
    ]
    choice_t1 = ["Normal", "Mildly Decreased"]
    df["T1_Ans"] = np.select(cond_t1, choice_t1, default="Severely Decreased")

    cond_t2 = [
        (df["Serum Creatinine"] <= q1),
        (df["Serum Creatinine"] > q1) & (df["Serum Creatinine"] <= q2),
        (df["Serum Creatinine"] > q2) & (df["Serum Creatinine"] <= q3),
        (df["Serum Creatinine"] > q3)
    ]
    choice_t2 = ["Very Low", "Low", "High", "Very High"]
    df["T2_Ans"] = np.select(cond_t2, choice_t2, default="")

    df["T3_Ans"] = (df["BMI"] > mediana_bmi).astype(int)

    df["T4_Ans"] = df["T Stage"].map(t_stage_counts).fillna(0).astype(int)

    return df

dfTrain = process_tasks(dfTrain)
dfTest = process_tasks(dfTest)

X_train = dfTrain.drop(["ID", "Status"], axis=1)
y_train = (dfTrain["Status"] == "Dead").astype(int)

X_test = dfTest.drop(["ID"], axis=1)

# ================================
# PREPROCESARE CATEGORICE (NATIV)
# ================================

cat_cols = X_train.select_dtypes(include=["object", "string"]).columns

for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")


model = HistGradientBoostingClassifier(
    random_state=42,
    max_iter=1500,
    early_stopping=True,
    class_weight="balanced"
)

param_list = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 50],
    "l2_regularization": [0, 0.5, 1]
}

tuner = RandomizedSearchCV(
    model,
    param_list,
    scoring="precision",
    n_iter=10,
    cv=5,
    n_jobs=-1,
    random_state=42
)

tuner.fit(X_train, y_train)
best = tuner.best_estimator_

print(best.predict_proba(X_train))

prag = 0.1
pred_proba = best.predict_proba(X_test)[:, 1] # extragem probabilitatea pt clasa 1 ("Dead")
pred = (pred_proba >= prag).astype(int)

# ==========
# Submission
# ==========

answers = []

# Folosim zip exact cum îți place pentru eficiență
for dID, t1, t2, t3, t4, t5 in zip(dfTest["ID"], dfTest["T1_Ans"], dfTest["T2_Ans"], dfTest["T3_Ans"], dfTest["T4_Ans"], pred):
    answers.append({
        "subtaskID": 1,
        "datapointID": dID,
        "answer": t1
    })
    answers.append({
        "subtaskID": 2,
        "datapointID": dID,
        "answer": t2
    })
    answers.append({
        "subtaskID": 3,
        "datapointID": dID,
        "answer": t3
    })
    answers.append({
        "subtaskID": 4,
        "datapointID": dID,
        "answer": t4
    })
    answers.append({
        "subtaskID": 5,
        "datapointID": dID,
        "answer": "Dead" if t5 else "Alive"
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index=False)