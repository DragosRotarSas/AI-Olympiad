import pandas as pd
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

dfTrain = pd.read_csv("train_data.csv")
dfTest = pd.read_csv("test_data.csv")

# ======
# Task 1
# ======

def numberOfPixels(img):
    return np.count_nonzero(img)

xrays_img = cv2.imread("xrays.png", cv2.IMREAD_GRAYSCALE)
Task1 = numberOfPixels(xrays_img)

# ======
# Task 2
# ======

def eLinie(x):
    return x[4] > 10

def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image=binary)

    nrLinii = 0
    for insula in stats[1:]:
        if eLinie(insula):
            nrLinii += 1

    nrPuncte = (retval - 1) - nrLinii

    return [nrPuncte, nrLinii]

# --- Extragere Feature-uri ---

X_train = []
y_train = []

for _, row in dfTrain.iterrows():
    img = cv2.imread("train/" + row["file_name"])
    f = getFeatures(img)
    X_train.append(f)
    y_train.append(row["label"])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

X_test = []
for _, row in dfTest.iterrows():
    img = cv2.imread("test/" + row["file_name"])
    f = getFeatures(img)
    X_test.append(f)

X_train = np.array(X_train)
X_test = np.array(X_test)

rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Cei mai buni parametri RF:", grid_search.best_params_)
print("Cel mai bun scor RF: ", grid_search.best_score_)

best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
pred_rf_encoded = best_rf.predict(X_test)
pred_rf = label_encoder.inverse_transform(pred_rf_encoded)

# Setam parametrul loss_function la MultiClass
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    verbose=0,
    random_state=42
)

cat_model.fit(X_train, y_train)
pred_cat_encoded = cat_model.predict(X_test)

# CatBoost returneaza o matrice columnara (ex: [[0], [2], [1]])
# label_encoder.inverse_transform asteapta un vector plat ([0, 2, 1])
# Folosim .flatten() pentru a-l repara
pred_cat = label_encoder.inverse_transform(pred_cat_encoded.flatten())


# ==========
# Submission
# ==========

answers = []

answers.append({
    "subtaskID": 1,
    "datapointID": 1,
    "answer": Task1
})

for pID, p in zip(dfTest["datapointID"], pred_cat):
    answers.append({
        "subtaskID": 2,
        "datapointID": pID,
        "answer": p
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index=False)