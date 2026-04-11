import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

# ======
# TASK 1
# ======

counter = {}

for i in dfTrain["enginetype"]:
    if i in counter:
        counter[i] += 1
    else:
        counter[i] = 1

valMaxim = 0
Task1 = ""

for i in counter:
    if valMaxim < counter[i]:
        valMaxim = counter[i]
        Task1 = i

Task1P = dfTrain["enginetype"].value_counts().idxmax()

# ======
# TASK 2
# ======

selectie = []

for row in dfTrain.itertuples():
    if row.fueltype == "gas":
        selectie.append(row.price)

Task2 = round(np.mean(selectie), 2)

Task2P = round(dfTrain[dfTrain["fueltype"] == "gas"]["price"].mean(), 2)

# ------
# TASK 3
# ------

# =======================================
# Pastram doar brandul din numele masinii
# =======================================

# .str.split() -> lista de cuvinte din propozitie, .str[0] -> primul cuvant
dfTrain["brand"] = dfTrain["CarName"].str.split().str[0]
dfTest["brand"]  = dfTest["CarName"].str.split().str[0]

dfTrain.drop("CarName", axis=1, inplace=True)
dfTest.drop("CarName", axis=1, inplace=True)

#print(dfTrain["brand"].unique())           # toate variabilele
#print(dfTrain["brand"].value_counts())     # de cate ori apare fiecare

# ============================
# Umplem locurile goale din db
# ============================

alese = [c for c in dfTrain.columns if c != "price"]

for col in alese:
    if is_numeric_dtype(dfTrain[col]):
        valFill = dfTrain[col].median()

    else:
        valFill = dfTrain[col].mode()[0]

    dfTrain[col] = dfTrain[col].fillna(valFill)
    dfTest[col] = dfTest[col].fillna(valFill)

#print(dfTrain.isna().sum())
#print(dfTest.isna().sum())

# ========================
# One hot encoding + model
# ========================

X_train = dfTrain.drop(["price", "CarID"], axis=1)
y_train = dfTrain["price"]
X_test  = dfTest.drop(["CarID"], axis=1)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test  = pd.get_dummies(X_test, drop_first=True)

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

#model = LinearRegression()
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

#print(pred)

# ==========
# Submission
# ==========

rows = []

# Task 1
rows.append({
    "subtaskID":1,
    "datapointID":1,
    "answer":Task1
})

rows.append({
    "subtaskID":2,
    "datapointID":1,
    "answer":Task2
})

for cID, i in zip(dfTest["CarID"], pred):
    rows.append({
        "subtaskID": 3,
        "datapointID": cID,
        "answer": i
    })

submission = pd.DataFrame(rows, columns=["subtaskID", "datapointID", "answer"])
submission.to_csv("submission.csv", index=False)

#print(dfTest.isnull().sum())
#print(pd.get_dummies(dfTest).dtypes)