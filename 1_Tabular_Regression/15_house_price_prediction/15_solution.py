import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

media = dfTrain["Square_Footage"].mean()

def getFeatures(df):
    df["Total_Size"] = df["Square_Footage"] + df["Garage_Size"] + df["Lot_Size"]
    df["Garage_to_Room_Ratio"] = df["Garage_Size"] / df["Total_Rooms"]
    df["Env_Stability_Index"] = ((df["Solar_Exposure_Index"] - df["Vibration_Level"])
                                 / df["Magnetic_Field_Strength"])

    df["Abs"] = abs(df["Square_Footage"] - media)

getFeatures(dfTrain)
getFeatures(dfTest)

Task1 = dfTest["Total_Size"]
Task2 = dfTest["Garage_to_Room_Ratio"]
Task3 = dfTest["Env_Stability_Index"]
Task4 = dfTest["Abs"]

# ======
# Task 5
# ======

test = ["House_Orientation_Angle", "Magnetic_Field_Strength", "Vibration_Level", "Env_Stability_Index", "Abs"]

X_train = dfTrain.drop(["ID", "Price"], axis=1)
y_train = dfTrain["Price"]
X_test = dfTest.drop(["ID"], axis=1)

#X_train = X_train.drop(test, axis=1)
#X_test = X_test.drop(test, axis=1)

model = HistGradientBoostingRegressor(
    random_state=42,
    loss="absolute_error",
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

param_list = {
    "l2_regularization": [0, 1]
}

tuner = RandomizedSearchCV(
    model,
    param_list,
    scoring="neg_mean_absolute_error",
    cv=10,
    n_iter=10,
    n_jobs=-1,
    random_state=42
)

tuner.fit(X_train, y_train)

best_HGB = tuner.best_estimator_
pred = best_HGB.predict(X_test)

# ==========
# Submission
# ==========

answers = []

for dID, t1, t2, t3, t4, t5 in zip(dfTest["ID"], Task1, Task2, Task3, Task4, pred):
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
        "answer": t5
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index=False)

"""
param_list = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63], # controlează complexitatea/overfitting-ul
    "min_samples_leaf": [20, 50],
    "l2_regularization": [0, 0.5, 1]
}
"""