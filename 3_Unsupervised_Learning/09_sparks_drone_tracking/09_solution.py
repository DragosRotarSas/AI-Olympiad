import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np

df = pd.read_csv("dataset.csv")

# Sortăm datele cronologic (foarte important pentru a urmări dronele corect)
df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'],
            #Jan 26, 2026, 12:00:03 AM
        format='%b %d, %Y, %I:%M:%S %p')

df = df.sort_values('Timestamp_dt')

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
prev_centers = None
timestamps_sec = []

# Extragem timpul de început pentru a calcula diferențele în secunde
t0 = pd.to_datetime(df["Timestamp"].unique()[0])

for ts in df["Timestamp"].unique():

    elem = []

    X = np.array(df[df["Timestamp"] == ts]["X"])
    Y = np.array(df[df["Timestamp"] == ts]["Y"])

    for i in range(len(X)):
        elem.append([X[i], Y[i]])

    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    model.fit(elem)
    curr_centers = np.array(model.cluster_centers_)

    # Partea nouă: Tracking-ul. Legăm noii centroizi de cei vechi pentru a păstra ordinea (ID-urile)
    if prev_centers is None:
        ordered_centers = curr_centers
    else:
        # linear_sum_assignment găsește potrivirea optimă minimizând distanța
        cost_matrix = np.linalg.norm(curr_centers[:, None] - prev_centers[None, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        ordered_centers = np.zeros_like(curr_centers)
        for r, c in zip(row_ind, col_ind):
            ordered_centers[c] = curr_centers[r]

    prev_centers = ordered_centers
    centers.append(ordered_centers)

    # Salvăm și secundele scurse pentru predicția de la Task 4
    current_time = pd.to_datetime(ts)
    timestamps_sec.append((current_time - t0).total_seconds())

centers = np.array(centers)

# ---- Task 4 ----

future_time = pd.to_datetime("Jan 26, 2026, 12:08:20 AM")
future_sec = (future_time - t0).total_seconds()

future_centers = []

for i in range(5):
    # Luăm istoricul pe axele X și Y pentru drona 'i'
    x_history = centers[:, i, 0]
    y_history = centers[:, i, 1]

    # Găsim ecuația dreptei folosind regresia liniară (viteză constantă)
    px = np.polyfit(timestamps_sec, x_history, 1)
    py = np.polyfit(timestamps_sec, y_history, 1)

    # Estimăm poziția în viitor
    pred_x = np.polyval(px, future_sec)
    pred_y = np.polyval(py, future_sec)

    future_centers.append([pred_x, pred_y])

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
        # Folosim 3 zecimale pentru mai multă precizie la evaluare
        answer = (str(round(pred[i][0], 3)) + "|" + str(round(pred[i][1], 3)))
        rows.append({
            "id": ts + "|" + str(i),
            "subtaskID": "task3",
            "answer": answer
        })

for i in range(5):
    answer_task4 = (str(round(future_centers[i][0], 3)) + "|" + str(round(future_centers[i][1], 3)))
    rows.append({
        "id": "Jan 26, 2026, 12:08:20 AM|" + str(i),
        "subtaskID": "task4",
        "answer": answer_task4
    })

submission = pd.DataFrame(rows, columns=["id", "subtaskID", "answer"])
submission.to_csv("submission.csv", index=False)