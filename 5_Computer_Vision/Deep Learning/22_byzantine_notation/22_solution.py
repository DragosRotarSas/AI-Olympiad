import pandas as pd
import numpy as np
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import LabelEncoder

import tqdm
import matplotlib.pyplot as plt
import cv2

SEED = 42
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
le = LabelEncoder()

# ==========================
# DATA, DATASET & DATALOADER
# ==========================

# PREPROCESS IMAGE FOR CNN AND TURN IT INTO A TENSOR
def getPreprocessedImageTensor(img):

    _, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    img = cv2.resize(img, (48, 48))

    img = img.astype(np.float32) / 255.0

    img = torch.tensor(img, dtype=torch.float)
    img = img.unsqueeze(0)

    return img

# IMAGE DATASET
class ImgDataSet(Dataset):
    def __init__(self, X_data, y_data = None):
        self.X = X_data
        if y_data is not None:
            self.y = y_data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.X[idx], cv2.IMREAD_GRAYSCALE)
        img_tensor = getPreprocessedImageTensor(img)
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype = torch.long)
            return img_tensor, label
        return img_tensor

# TRAINING DATA
dfTrain = pd.read_csv("dataset_train.csv")

dfTrain["Effect"] = dfTrain["Effect"].astype(str).str.strip()
dfTrain["Effect"] = le.fit_transform(dfTrain["Effect"])

train_dataset = ImgDataSet(np.array(dfTrain["Path"]), np.array(dfTrain["Effect"]))

train_dataloader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

val_dataloader = None

"""
When training

TRAIN_SIZE = int(0.8 * len(dfTrain))

X_train = np.array(dfTrain["Path"])[:TRAIN_SIZE]
X_val = np.array(dfTrain["Path"])[TRAIN_SIZE:]
y_train = np.array(dfTrain["Effect"])[:TRAIN_SIZE]
y_val = np.array(dfTrain["Effect"])[TRAIN_SIZE:]

train_dataset = ImgDataSet(X_train, y_train)
val_dataset = ImgDataSet(X_val, y_val)

train_dataloader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False
)
"""

# =====
# MODEL
# =====

class MyCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels = num_channels,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2
            ),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.LazyLinear(num_classes),
            # nn.Softmax() CrossEntropy are deja SoftMax
        )

    def forward(self, x):
        return self.network(x)



model = MyCNN(1, len(le.classes_)).to(device)

# =============
# TRAINING LOOP
# =============

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

EPOCHS = 10

def trainLoop(isTraining = True):
    for epoch in tqdm.tqdm(range(EPOCHS)):

        model.train()
        loss_total_antrenament = 0

        for data, label in train_dataloader:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(data)

            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()

            loss_total_antrenament += loss.item()

        model.eval()
        loss_total_validare = 0

        if isTraining:
            with torch.inference_mode():

                for val_data, val_label in val_dataloader:
                    val_data = val_data.to(device)
                    val_label = val_label.to(device)

                    val_outputs = model(val_data)
                    val_loss = loss_fn(val_outputs, val_label)
                    loss_total_validare += val_loss

            print(f"\nEpoch: {epoch} Loss Antrenare: {loss_total_antrenament / len(train_dataloader)} Loss Validare: {loss_total_validare / len(val_dataloader)}")

trainLoop(isTraining = False)

# ================
# TEST PREDICTIONS
# ================

def getSequenceImages(SEQ_PATH):
    img = cv2.imread(SEQ_PATH, cv2.IMREAD_GRAYSCALE)
    _, img_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    all_contours, _ = cv2.findContours(
        image=img_thresh,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    obiecte = []
    for c in all_contours:
        if cv2.contourArea(c) > 20:
            x, y, w, h = cv2.boundingRect(c)
            crop = img[y:y + h, x:x + w]

            latura_maxima = max(w, h)
            patrat_alb = np.full((latura_maxima, latura_maxima), 255, dtype=np.uint8)

            x_offset = (latura_maxima - w) // 2
            y_offset = (latura_maxima - h) // 2

            patrat_alb[y_offset:y_offset + h, x_offset:x_offset + w] = crop

            obiecte.append((x, patrat_alb))

    obiecte.sort(key=lambda obj: obj[0])

    secventa_finala = [obj[1] for obj in obiecte]

    return secventa_finala

def getSequence(SEQ_PATH):

    neume = getSequenceImages(SEQ_PATH)
    pred = []

    model.eval()
    with torch.inference_mode():
        for n in neume:

            n_tensor = getPreprocessedImageTensor(n)
            tensor_n = n_tensor.unsqueeze(0).to(device)

            output = model(tensor_n)
            clasa_prezisa = torch.argmax(output, dim=1).item()
            efect_real = le.inverse_transform([clasa_prezisa])[0]

            if efect_real in ["A", "B"]:
                valoare_numerica = 0
            else:
                valoare_numerica = int(efect_real)

            pred.append(valoare_numerica)

    rezultat_cumulat = []
    suma_curenta = 0

    for efect in pred:
        suma_curenta += efect
        rezultat_cumulat.append(str(suma_curenta))

    return "|".join(rezultat_cumulat)


dfTest = pd.read_csv("dataset_eval.csv")
pred = []

for i, row in dfTest.iterrows():
    pred.append(getSequence(row["datapointID"]))

# ==========
# SUBMISSION
# ==========

answers = []

for i, p in zip(dfTest["datapointID"], pred):
    answers.append({
        "subtaskID": 1,
        "datapointID": i,
        "answer": p
    })

submission = pd.DataFrame(answers)
submission.to_csv("submission.csv", index = False)