import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("outputs/wesad_features.csv")

feature_cols = [
    "eda_mean", "eda_std", "eda_min", "eda_max",
    "temp_mean", "temp_std", "temp_min", "temp_max"
]

X = df[feature_cols]
y = df["label"]
groups = df["subject_id"]

logo = LeaveOneGroupOut()
fold_rows = []
all_true = []
all_pred = []

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    true = y.iloc[test_idx]

    fold_rows.append({
        "held_out_subject": groups.iloc[test_idx].iloc[0],
        "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
        "stress_recall": recall_score(true, pred, pos_label=1, zero_division=0)
    })

    all_true.extend(true.tolist())
    all_pred.extend(pred.tolist())

results = pd.DataFrame(fold_rows)
results.to_csv("outputs/logreg_fold_results.csv", index=False)

print(results)
print("\nMean fold scores:")
print(results[["macro_f1", "balanced_accuracy", "stress_recall"]].mean())

print("\nPooled confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(all_true, all_pred, labels=[0, 1]))