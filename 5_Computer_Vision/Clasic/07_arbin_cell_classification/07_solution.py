import cv2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")
cellSize = 64

# =================================
# Obtinere feature-uri pentru train
# =================================

def get_cell_features(cell):

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # resize
    small = cv2.resize(gray, (32, 32))

    # threshold
    _, binary = cv2.threshold(small, 100, 255, cv2.THRESH_BINARY)

    features = binary.flatten()
    return features

X_train = []
y_train = []

for _, row in dfTrain.iterrows():

    img = cv2.imread(row["id"])
    cell = img[0:cellSize, 0:cellSize]

    features = get_cell_features(cell)

    X_train.append(features)
    y_train.append(row["type"])

    # augmentare rotire
    for angle in [-10, 10]:

        M = cv2.getRotationMatrix2D((32,32), angle, 1)

        rotated = cv2.warpAffine(
            cv2.resize(cell,(64,64)),
            M,
            (64,64)
        )

        features = get_cell_features(rotated)

        X_train.append(features)
        y_train.append(row["type"])

X_train = np.array(X_train)
y_train = np.array(y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)

# ========================
# Antrenare model pe train
# ========================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_pca, y_train)

# =============================
# Transformare imagini in linii
# =============================

linii = []

for _, row in dfTest.iterrows():
    img = cv2.imread(row["id"])
    linie = []

    for r in range(4):
        for c in range(4):
            cell = img[
                r * cellSize:(r + 1) * cellSize,
                c * cellSize:(c + 1) * cellSize
            ]
            features = get_cell_features(cell)
            f_2d = features.reshape(1, -1)
            f_scaled = scaler.transform(f_2d)
            f_pca = pca.transform(f_scaled)

            predCell = model.predict(f_pca)
            linie.append(int(predCell[0]))
    linii.append(linie)

# ==========
# Submission
# ==========

rows = []

for linie, iID in zip(linii, dfTest["id"]):
    rows.append({
        "subtaskID": 1,
        "ID": iID,
        "answer": linie[0]
    })

    nr = 0
    for i in range(len(linie)):
        if linie[i] == 1 :
            nr = nr + pow(2, 16 - i - 1)
    if nr >= 32768:
        nr -= 65536

    rows.append({
        "subtaskID": 2,
        "ID": iID,
        "answer": nr
    })

submission = pd.DataFrame(rows)
submission.to_csv("submission.csv", index=False)