import pandas as pd
from catboost import Pool, FeaturesData, CatBoostRegressor

dfTrain = pd.read_csv("train_data.csv")
dfTest = pd.read_csv("test_data.csv")

# ---- Task 1 ----
Task1 = (dfTest[dfTest["Weather"] == "Fog"]["City A"] == "Barlad").sum()

# ====
# Data
# ====

X_train = dfTrain.drop(["ID", "deliver_time"], axis = 1)
y_train = dfTrain["deliver_time"]
X_test = dfTest.drop(["ID"], axis = 1)

def getFeatures(X):
    #X["Drum"] = X["City A"] + "_" + X["City B"]
    return X

X_train = getFeatures(X_train)
X_test = getFeatures(X_test)

# =====
# Model
# =====

cat_cols = X_train.select_dtypes(include=["string", "object"]).columns
cat_features_indices = list(cat_cols)

train_data = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_features_indices
)

test_data = Pool(
    data=X_test,
    cat_features=cat_features_indices
)

model = CatBoostRegressor(
    iterations = 1000,
    learning_rate = 0.1,
    loss_function = "MAE",
    random_state = 42
)

model.fit(train_data)
pred = model.predict(test_data)

# ==========
# Submission
# ==========

answers = []

answers.append({
    "subtaskID": 1,
    "datapointID": 1,
    "answer": Task1
})

for i, p in zip(dfTest["ID"], pred):
    answers.append({
        "subtaskID": 2,
        "datapointID": i,
        "answer": p
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index = False)