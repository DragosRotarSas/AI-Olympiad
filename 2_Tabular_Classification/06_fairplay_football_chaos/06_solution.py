import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

AGGRESSIVE_STYLES = ["AggressiveTackler", "RiskTaker", "HighPressure", "ChaosInducer"]

def build_features(df):
    df = df.copy()

    # ---- TeamStyles features ----
    style_aggression_score = []
    aggression_styles_counter = []

    for i, row in df.iterrows():
        styles = row["TeamStyles"].split(", ")
        ct = 0

        for style in styles:
            if style in AGGRESSIVE_STYLES:
                ct += 1

        aggression_styles_counter.append(ct)
        style_aggression_score.append(ct / len(styles) if len(styles) > 0 else 0)

    df["StyleAggressionScore"] = style_aggression_score
    df["AggressionStylesCounter"] = aggression_styles_counter

    # ---- Safe ratios ----
    goals_to_shots_ratio = []
    card_to_shot_ratio = []
    red_ratio = []
    yellow_ratio = []

    for i, row in df.iterrows():
        shots = row["Shots"]
        total = row["TotalCards"]
        goals = row["Goals"]
        red = row["RedCards"]
        yellow = row["YellowCards"]

        if shots == 0:
            goals_to_shots_ratio.append(0)
            card_to_shot_ratio.append(0)
        else:
            goals_to_shots_ratio.append(goals / shots)
            card_to_shot_ratio.append(total / shots)

        if total == 0:
            red_ratio.append(0)
            yellow_ratio.append(0)
        else:
            red_ratio.append(red / total)
            yellow_ratio.append(yellow / total)

    df["GoalsToShotsRatio"] = goals_to_shots_ratio
    df["CardToShotRatio"] = card_to_shot_ratio
    df["RedCardRatio"] = red_ratio
    df["YellowCardRatio"] = yellow_ratio

    # ---- Dummies din TeamStyles ----
    df = df.join(df["TeamStyles"].str.get_dummies(sep=", "))

    # ---- Pereche echipe, independent de ordine ----
    teams = []
    for i, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        if home > away:
            teams.append(away + "_" + home)
        else:
            teams.append(home + "_" + away)

    df["Teams"] = teams

    return df


dfTrain = build_features(dfTrain)
dfTest = build_features(dfTest)

# ---- Task 1 ----
Task1 = (dfTest["HomeTeam"] == "Chelsea").sum() + (dfTest["AwayTeam"] == "Chelsea").sum()

# ---- Task 2 ----
Task2 = dfTest["StyleAggressionScore"]

# ---- Task 3 ----
X_train = dfTrain.drop(["MatchID", "TeamStyles", "chaos_label"], axis=1)
y_train = dfTrain["chaos_label"]
X_test = dfTest.drop(["MatchID", "TeamStyles"], axis=1)

num_cols = X_train.select_dtypes(include=["number"]).columns
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns

pipe_num = SimpleImputer(strategy="median")
pipe_cat = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(transformers=[
    ("num", pipe_num, num_cols),
    ("cat", pipe_cat, cat_cols)
])

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

model = GradientBoostingClassifier(
    n_estimators=1000,
    random_state=42,
    learning_rate=0.1
)

clf = Pipeline(steps=[
    ("prep", preprocess),
    ("model", model)
])

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

#f1 = f1_score(y_val, pred, average="macro")
#print(f1)

# ---- Submission ----
rows = []

rows.append({
    "subtaskID": 1,
    "datapointID": 1,
    "answer": Task1
})

for sID, t2, t3 in zip(dfTest["MatchID"], Task2, pred):
    rows.append({
        "subtaskID": 2,
        "datapointID": sID,
        "answer": round(float(t2), 2)
    })
    rows.append({
        "subtaskID": 3,
        "datapointID": sID,
        "answer": t3
    })

submission = pd.DataFrame(rows, columns=["subtaskID", "datapointID", "answer"])
submission.to_csv("submission.csv", index=False)
