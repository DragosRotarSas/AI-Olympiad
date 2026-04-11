import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

airport_delay_mean = dfTrain.groupby("airport")["delay"].mean()
media_generala_delay = airport_delay_mean.mean()

def getFeatures(df):
    # 1. Calculăm mediile (Target Encoding)
    df["media_aeroportului"] = df["airport"].map(airport_delay_mean)
    df["media_aeroportului"] = df["media_aeroportului"].fillna(media_generala_delay)

    # 2. Rate (Feature Engineering)
    # Folosim .divide ca să putem pune fill_value=0 direct
    df["rate_del15"] = df["arr_del15"] / df["arr_flights"]
    df["rate_cancelled"] = df["arr_cancelled"] / df["arr_flights"]

    # 3. FIX: Tratăm NaNs separat pe tipuri de date
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0).fillna(0)

    # Optional: tratăm textul dacă are NaNs
    obj_cols = df.select_dtypes(include=['object', 'string']).columns
    df[obj_cols] = df[obj_cols].fillna("Missing")

    return df

dfTrain = getFeatures(dfTrain)
dfTest = getFeatures(dfTest)

#dfTrain = dfTrain.dropna(subset=["delay"])

dfTrain["delay"] = dfTrain["delay"].fillna(
    dfTrain["airport"].map(airport_delay_mean)
).fillna(media_generala_delay)

X_train = dfTrain.drop(["sample_id", "carrier_name", "airport", "airport_name", "delay"], axis=1)
y_train = dfTrain["delay"]
X_test = dfTest.drop(["sample_id", "carrier_name", "airport", "airport_name"], axis=1)

cat_cols = X_train.select_dtypes(include=["string", "object"]).columns

for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

model = HistGradientBoostingRegressor(
    random_state=42,
    max_iter=1500,
    early_stopping=True,
    loss="absolute_error"
)

param_list = {
    "l2_regularization": [0, 0.5, 1],
    "learning_rate": [0.1, 0.05],
    "max_leaf_nodes": [15, 31, 60]
}

tuner = RandomizedSearchCV(
    model,
    param_list,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    cv=5
)

tuner.fit(X_train, y_train)
best = tuner.best_estimator_
pred = best.predict(X_test)

answers = []

for sID, p in zip(dfTest["sample_id"], pred):
    answers.append({
        "sample_id": sID,
        "delay": p
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index=False)
