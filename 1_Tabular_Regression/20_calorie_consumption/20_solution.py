import pandas as pd
import numpy as np
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


dfTrain = pd.read_csv("train_data.csv")
dfTest = pd.read_csv("test_data.csv")

# =========
# TASKS 1-4
# =========

Task1 = len(dfTrain)
Task2 = (dfTrain["Gender"] == "male").sum()
Task3 = dfTrain["Duration"].mean()
Task4 = (dfTrain["Age"] >= 75).sum()

# ==========
# TASK 5 & 6
# ==========

X_train = dfTrain.drop(["User_ID", "Calories"], axis = 1)
y_train = dfTrain["Calories"]
X_test = dfTest.drop(["Subtask", "User_ID"], axis = 1)

# Encoding
encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=np.nan
)
X_train_encoded = X_train.copy()
X_train_encoded[["Gender"]] = encoder.fit_transform(X_train[["Gender"]])

# Imputare
imputer = KNNImputer(n_neighbors = 10)
X_train_imputed_raw = imputer.fit_transform(X_train_encoded)

# Back 2 normal
X_train_imputed = pd.DataFrame(X_train_imputed_raw, columns=X_train.columns)
X_train_imputed[['Gender']] = encoder.inverse_transform(X_train_imputed[['Gender']].round().astype(int))

# Acum la fel pe test
X_test_encoded = X_test.copy()
X_test_encoded[["Gender"]] = encoder.transform(X_test[["Gender"]])
X_test_imputed_raw = imputer.transform(X_test_encoded)
X_test_imputed = pd.DataFrame(X_test_imputed_raw, columns = X_test.columns)
X_test_imputed[["Gender"]] = encoder.inverse_transform(X_test_imputed[["Gender"]].round().astype(int))

cat_cols = list(X_train_imputed.select_dtypes(include = ["object", "string"]).columns)

train_pool = Pool(
    data=X_train_imputed,
    label=y_train,
    cat_features=cat_cols
)

test_pool = Pool(
    data = X_test_imputed,
    cat_features = cat_cols
)

model = CatBoostRegressor(
    iterations=10000,
    loss_function="MAE",
#    early_stopping_rounds=50,
    random_state=42,
    nan_mode='Max',
    random_strength=5,
    depth = 8,
    learning_rate = 0.03,
    l2_leaf_reg = 1,
    verbose = 100
)

"""
grid = {
    'depth': [4, 6, 8],                  # Adancimea arborelui. Peste 8 dureaza mult si tinde spre overfitting
    'learning_rate': [0.01, 0.03, 0.1],  # Pasii de invatare. Combinam pasi mici cu pasi mai mari
    'l2_leaf_reg': [1, 3, 5, 9],         # Regularizarea L2
    'random_strength': [1, 5]            # Adauga zgomot la taieturi pentru a preveni memorarea datelor
}

rezultate = model.randomized_search(
    param_distributions = grid,
    X=train_pool,
    cv=3,
    n_iter=10,
    verbose=1000
)

print("Cei mai buni parametri:")
print(rezultate['params'])
"""

model.fit(train_pool)
pred = model.predict(test_pool)
dfTest["pred"] = pred

# ==========
# SUBMISSION
# ==========

answers = []

answers.append({
    "subtaskID": 1,
    "datapointID": 1,
    "answer": Task1
})

answers.append({
    "subtaskID": 2,
    "datapointID": 1,
    "answer": Task2
})

answers.append({
    "subtaskID": 3,
    "datapointID": 1,
    "answer": Task3
})

answers.append({
    "subtaskID": 4,
    "datapointID": 1,
    "answer": Task4
})

for i, p in zip(dfTest[dfTest["Subtask"] == 5]["User_ID"], dfTest[dfTest["Subtask"] == 5]["pred"]):
    answers.append({
        "subtaskID": 5,
        "datapointID": i,
        "answer": p
    })

for i, p in zip(dfTest[dfTest["Subtask"] == 6]["User_ID"], dfTest[dfTest["Subtask"] == 6]["pred"]):
    answers.append({
        "subtaskID": 6,
        "datapointID": i,
        "answer": p
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index = False)