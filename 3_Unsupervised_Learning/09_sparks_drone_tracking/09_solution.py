import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

df = pd.read_csv("dataset.csv")
s = 125

# ---- Task 1 ----
Task1 = len(df["Name"].unique())

# ---- Task 2 ----
"""
for name in df["Name"].unique():
    array = np.array(df[df["Name"] == name]["X"].astype(int))
    array.sort()
    print(array)
"""
Task2 = 5

# ---- Task 3 ----

centers = []

for ts in df["Timestamp"].unique():

    elem = []
    
    X = np.array(df[df["Timestamp"] == ts]["X"])
    Y = np.array(df[df["Timestamp"] == ts]["Y"])
    
    for i in range(len(X)):
        elem.append([X[i], Y[i]])
        
    model = KMeans(
        n_clusters=5
    )
    
    model.fit(elem)
    centers.append(np.array(model.cluster_centers_))
    
centers = np.array(centers)
    
# ---- Submission ----
rows = []

rows.append({
    "id": "GLOBAL",
    "subtaskID": "task1",
    "answer": Task1
})

rows.append({
    "id": "GLOBAL",
    "subtaskID": "task2",
    "answer": Task2
})

for ts, pred in zip(df["Timestamp"].unique(), centers):
    for i in range(5):
        answer = (str(round(pred[i][0], 2)) + "|" + str(round(pred[i][1], 2)))
        rows.append({
            "id": ts + "|" + str(i),
            "subtaskID": "task3",
            "answer": answer
        })


for i in range(5):
    rows.append({
        "id": "Jan 26, 2026, 12:08:20 AM|" + str(i),
        "subtaskID": "task4",
        "answer": "0|0"
    })    

submission = pd.DataFrame(rows, columns=["id", "subtaskID", "answer"])
submission.to_csv("submission.csv", index = False)