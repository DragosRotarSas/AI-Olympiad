import pandas as pd
import calendar
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier

dfTrain = pd.read_csv("dataset_train.csv")
dfEval = pd.read_csv("dataset_eval.csv")

# TASK 1
dfTrain["Distance"] = dfTrain["Distance"].astype(float)
dfTrain["Elapsed Time"] = dfTrain["Elapsed Time"].astype(float)

dfTrain["kmH"] = dfTrain["Distance"] / (dfTrain["Moving Time"] / 3600.0)

dfTrain["Activity Date"] = pd.to_datetime(
    dfTrain["Activity Date"],
    format="%b %d, %Y, %I:%M:%S %p"
)

Task1 = dfTrain.groupby(dfTrain["Activity Date"].dt.month)["kmH"].mean()

# TASK 2
def add_features_no_haversine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Distance"] = df["Distance"].astype(float)
    df["Elapsed Time"] = df["Elapsed Time"].astype(float)
    df["Moving Time"] = df["Moving Time"].astype(float)

    df["speed_kmh"] = df["Distance"] / (df["Elapsed Time"] / 3600.0)
    df["moving_speed_kmh"] = df["Distance"] / (df["Moving Time"] / 3600.0)
    df["moving_ratio"] = df["Moving Time"] / df["Elapsed Time"]

    lat1 = df["Starting Latitude"].astype(float)
    lon1 = df["Starting Longitude"].astype(float)
    lat2 = df["Finish Latitude"].astype(float)
    lon2 = df["Finish Longitude"].astype(float)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    df["straight_proxy"] = np.sqrt(dlat**2 + dlon**2)
    denom = df["straight_proxy"].replace(0, np.nan)
    df["detour_ratio"] = (df["Distance"] / denom).fillna(50.0).clip(upper=50.0)

    return df

dfTrainF = add_features_no_haversine(dfTrain)
dfEvalF = add_features_no_haversine(dfEval)

features = [
    "Distance", "Elapsed Time", "Moving Time",
    "speed_kmh", "moving_speed_kmh", "moving_ratio",
    "straight_proxy", "detour_ratio"
]

X_train = dfTrainF[features]
y_train = dfTrainF["Label"]
X_eval = dfEvalF[features]

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_eval)

# Submission

rows = []

for month, mean_kmh in Task1.items():
    rows.append({
        "subtaskID": 1,
        "Answer1": calendar.month_abbr[month],
        #"Answer2": round(mean_kmh, 5)
        "Answer2": math.trunc(mean_kmh * (10 ** 5)) / (10 ** 5)
    })

for ride_id, label in zip(dfEval["Activity ID"], pred):
    rows.append({
        "subtaskID": 2,
        "Answer1": ride_id,
        "Answer2": label
    })

submission = pd.DataFrame(rows, columns=["subtaskID", "Answer1", "Answer2"])
submission.to_csv("submission.csv", index=False)
