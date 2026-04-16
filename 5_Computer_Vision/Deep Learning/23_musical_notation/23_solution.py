import pandas as pd
import numpy as np
import cv2
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, InterpolationMode

from sklearn.preprocessing import LabelEncoder

import tqdm
import matplotlib.pyplot as plt

# ==================
# SEQUENCE FUNCTIONS
# ==================

SEED = 32
BATCH_SIZE = 32
torch.manual_seed(SEED)

empty_path = "train_data/empty_piano_score.png"
empty = cv2.imread(empty_path)

# GET LOCATION OF LINES
empty = empty[::, 25: len(empty[0])] # REMOVE START SIGN
_, empty = cv2.threshold(empty, 127, 255, cv2.THRESH_BINARY) # BINARY
pixel_randuri = []
for i in range(len(empty)):
    if empty[i][0][0] == 0:
        pixel_randuri.append(i)

def getCleanPartition(IMG_PATH, isTrain = False):
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if not isTrain:
        img = img[::, 25: len(img[0])]
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    for pixel in pixel_randuri:
        if pixel < len(img):
            img[pixel] = 255
    return img

def getObjectsFromCleanPartition(clean_partition):

    img_inverse = cv2.bitwise_not(clean_partition)

    contours, _ = cv2.findContours(
        img_inverse,
        mode = cv2.RETR_EXTERNAL,
        method = cv2.CHAIN_APPROX_SIMPLE
    )

    unverified_objects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        crop = clean_partition[::, x - 2: x + w + 2]
        unverified_objects.append([x - 2, x + w + 2, crop])

    unverified_objects_sorted = sorted(unverified_objects, key=lambda obj:obj[0])

    objects = []
    objects.append(unverified_objects_sorted[0])

    for i in range(1, len(unverified_objects_sorted)):
        # SAME OBJECT
        if objects[i-1][0] <= unverified_objects_sorted[i][0] <= objects[i-1][1]:
            #print(f"first: {objects[i-1][1]} second: {unverified_objects_sorted[i][1]}")
            objects[i-1][1] = unverified_objects_sorted[i][1]   # MAKE IT LONGER
            objects[i - 1][2] = clean_partition[::, objects[i-1][0]:objects[i-1][1]] # UPDATED IMAGE
            objects.append(objects[i-1])
        # DIFFERENT OBJECT
        else:
            objects.append(unverified_objects_sorted[i])

    objects_images = []
    objects_images.append(objects[0][2])

    for i in range(1, len(objects)):
        if objects[i] != objects[i - 1]:
            objects_images.append(objects[i][2])

    objects_images_resized = []
    for obj in objects_images:

        y_coords, x_coords = np.where(obj == 0)

        if len(y_coords) == 0:
            continue

        y_start, y_end = np.min(y_coords), np.max(y_coords)
        x_start, x_end = np.min(x_coords), np.max(x_coords)

        crop = obj[y_start:y_end + 1, x_start:x_end + 1]


        h_crop, w_crop = crop.shape

        latura_maxima = max(h_crop, w_crop)

        scale = 44.0 / latura_maxima

        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)

        crop_resized = cv2.resize(crop, (new_w, new_h))

        white_background = np.full((48, 48), 255, dtype=np.uint8)

        off_x = (48 - new_w) // 2
        off_y = (48 - new_h) // 2

        white_background[off_y:off_y + new_h, off_x:off_x + new_w] = crop_resized

        objects_images_resized.append(white_background)

    return objects_images_resized

# =============
# TRAINING DATA
# =============

TRAIN_IMG_PATH = "train_data/all_piano_notes_duration.png"
TRAIN_IMG = getCleanPartition(TRAIN_IMG_PATH, True)
TRAIN_IMAGES = getObjectsFromCleanPartition(TRAIN_IMG)
TRAIN_LABELS = [0, 1, 2, 3, 4]

TRAIN_DATA_TENSORS = []
TRAIN_LABEL_TENSORS = []

transforms = v2.Compose([
    v2.ToPILImage(),
    v2.RandomRotation(degrees = 5, interpolation=InterpolationMode.NEAREST, expand=False, fill=255),
    v2.RandomAffine(degrees = 0, translate = (0.1, 0.1), scale = (0.8, 1.2), shear=None, interpolation=InterpolationMode.NEAREST, fill = 255),
    v2.GaussianBlur(kernel_size = 3, sigma=(0.1, 2.0)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
])

for img, label in zip(TRAIN_IMAGES, TRAIN_LABELS):
    img_tensor = torch.tensor(img, dtype = torch.float).unsqueeze(0) / 255
    label_tensor = torch.tensor(label, dtype = torch.long)
    #print(f"normal: {img_tensor} shape: {img_tensor.shape} ")

    TRAIN_DATA_TENSORS.append(img_tensor)
    TRAIN_LABEL_TENSORS.append(label_tensor)

    for _ in range(100):
        TRAIN_DATA_TENSORS.append(transforms(img))
        TRAIN_LABEL_TENSORS.append(label_tensor)
        #print(f"transformed: {test} shape: {test.shape}")

TRAIN_DATA_BATCH = torch.stack(TRAIN_DATA_TENSORS)
TRAIN_LABEL_BATCH = torch.stack(TRAIN_LABEL_TENSORS)

"""
for obj in TRAIN_OBJECTS:
    plt.imshow(obj)
    plt.show()
"""


# =====
# MODEL
# =====

class MyCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.network = nn.Sequential (
            nn.Conv2d(num_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.LazyLinear(num_classes)
        )

    def forward(self, X):
        return self.network(X)

model = MyCNN(1, len(TRAIN_LABELS))

# =====
# TRAIN
# =====

EPOCHS = 50

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
model.train()

losses = []
for epoch in tqdm.tqdm(range(EPOCHS)):

    data = TRAIN_DATA_BATCH
    labels = TRAIN_LABEL_BATCH

    optimizer.zero_grad()
    pred = model(data)

    loss = loss_fn(pred, labels)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()



"""
for i, loss in enumerate(losses):
    print(f"Epoch: {i} Loss: {loss}")
"""

# ===========
# PREDICTIONS
# ===========

model.eval()

Task1 = []
Task2 = []
Task3 = []

with torch.inference_mode():
    for i in tqdm.tqdm(range(2000)):
        partition_path = f"test_data/{i}.png"
        clean_partition = getCleanPartition(partition_path)
        objects = getObjectsFromCleanPartition(clean_partition)

        Task1.append(len(objects))
        if len(objects) == 0:
            Task2.append(0)
            Task3.append("")
            continue

        objects_tensor = torch.tensor(np.array(objects), dtype=torch.float).unsqueeze(1) / 255
        pred_tensor = model(objects_tensor)
        pred_classes = torch.argmax(pred_tensor, dim=1).tolist()

        s = 0
        for p in pred_classes:
            if p == 0:
                s += 1
            elif p == 1:
                s += 1 / 2
            elif p == 2:
                s += 1 / 4
            elif p == 3:
                s += 1 / 8
            else:
                s += 1 / 16

        Task2.append(s)

        Task3.append("A-1/1") # Placeholder

answer = []

for i, t1 in zip(range(2000), Task1):
    answer.append({
        "subtaskID": 1,
        "datapointID": i,
        "answer": t1
    })

for i, t2 in zip(range(2000), Task2):
    answer.append({
        "subtaskID": 2,
        "datapointID": i,
        "answer": t2
    })

for i, t3 in zip(range(2000), Task3):
    answer.append({
        "subtaskID": 3,
        "datapointID": i,
        "answer": t3
    })

submission = pd.DataFrame(answer)
submission.to_csv("submission.csv", index = False)