from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier


# ============================================================
# 1. Paths
# ============================================================

DATA_FILE = "outputs/wesad_features.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Features
# ============================================================

FEATURE_COLS = [
    "eda_mean",
    "eda_std",
    "eda_min",
    "eda_max",
    "temp_mean",
    "temp_std",
    "temp_min",
    "temp_max"
]


# ============================================================
# 3. Load data
# ============================================================

df = pd.read_csv(DATA_FILE)

X = df[FEATURE_COLS]
y = df["label"]
groups = df["subject_id"]


# ============================================================
# 4. LOSO
# ============================================================

logo = LeaveOneGroupOut()

fold_results = []
all_predictions = []

y_true_all = []
y_pred_all = []


# ============================================================
# 5. Run XGBoost
# ============================================================

for fold_num, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    held_out_subject = groups.iloc[test_idx].iloc[0]

    # --------------------------------------------------------
    # Build a fresh model for every LOSO fold
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    stress_recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    fold_results.append({
        "model": "XGBoost",
        "held_out_subject": held_out_subject,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "stress_recall": stress_recall
    })

    # Store predictions
    y_true_all.extend(y_test.tolist())
    y_pred_all.extend(y_pred.tolist())

    for true_label, predicted_label in zip(
        y_test,
        y_pred
    ):
        all_predictions.append({
            "model": "XGBoost",
            "held_out_subject": held_out_subject,
            "true_label": int(true_label),
            "predicted_label": int(predicted_label)
        })

    print(
        f"Fold {fold_num:02d} | "
        f"Subject {held_out_subject} | "
        f"Macro-F1 = {macro_f1:.4f} | "
        f"Balanced Acc = {balanced_acc:.4f} | "
        f"Stress Recall = {stress_recall:.4f}"
    )


# ============================================================
# 6. Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true_all,
    y_pred_all,
    labels=[0, 1]
)

print("\nPooled confusion matrix:")
print(cm)


# ============================================================
# 7. Save fold results
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)

fold_results_df.to_csv(
    OUTPUT_DIR / "person3_xgboost_fold_results.csv",
    index=False
)


# ============================================================
# 8. Save predictions
# ============================================================

predictions_df = pd.DataFrame(
    all_predictions
)

predictions_df.to_csv(
    OUTPUT_DIR / "person3_xgboost_predictions.csv",
    index=False
)


# ============================================================
# 9. Summary statistics
# ============================================================

summary = (
    fold_results_df[
        [
            "macro_f1",
            "balanced_accuracy",
            "stress_recall"
        ]
    ]
    .agg(["mean", "std"])
)

summary.to_csv(
    OUTPUT_DIR / "person3_xgboost_summary.csv"
)


# ============================================================
# 10. Print final result
# ============================================================

print("\n====================================")
print("XGBOOST FINAL RESULTS")
print("====================================")

print(summary)