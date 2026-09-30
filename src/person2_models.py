from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
# 3. Load dataset
# ============================================================

df = pd.read_csv(DATA_FILE)

required_cols = FEATURE_COLS + [
    "label",
    "subject_id"
]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"Missing columns: {missing_cols}"
    )

X = df[FEATURE_COLS]
y = df["label"]
groups = df["subject_id"]


# ============================================================
# 4. LOSO setup
# ============================================================

logo = LeaveOneGroupOut()


# ============================================================
# 5. Model definitions
# ============================================================

def make_svm():

    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        )
    )


def make_random_forest():

    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


models = {
    "SVM": make_svm,
    "Random_Forest": make_random_forest
}


# ============================================================
# 6. Run LOSO experiments
# ============================================================

all_fold_results = []
all_predictions = []


for model_name, build_model in models.items():

    print("\n====================================")
    print(model_name)
    print("====================================")

    y_true_all = []
    y_pred_all = []

    for fold_num, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Create a fresh model for each fold
        model = build_model()

        # Train
        model.fit(
            X_train,
            y_train
        )

        # Predict
        y_pred = model.predict(X_test)

        # Subject held out in this fold
        held_out_subject = (
            groups.iloc[test_idx].iloc[0]
        )

        # -------------------------------
        # Metrics
        # -------------------------------

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

        all_fold_results.append({
            "model": model_name,
            "held_out_subject": held_out_subject,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
            "stress_recall": stress_recall
        })

        # Store pooled predictions
        y_true_all.extend(
            y_test.tolist()
        )

        y_pred_all.extend(
            y_pred.tolist()
        )

        # Store individual predictions
        for true_label, predicted_label in zip(
            y_test,
            y_pred
        ):

            all_predictions.append({
                "model": model_name,
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

    # ========================================================
    # Pooled confusion matrix
    # ========================================================

    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=[0, 1]
    )

    print("\nPooled confusion matrix:")
    print(cm)


# ============================================================
# 7. Save fold-level results
# ============================================================

fold_results = pd.DataFrame(
    all_fold_results
)

fold_results.to_csv(
    OUTPUT_DIR / "person2_fold_results.csv",
    index=False
)


# ============================================================
# 8. Save predictions
# ============================================================

predictions = pd.DataFrame(
    all_predictions
)

predictions.to_csv(
    OUTPUT_DIR / "person2_predictions.csv",
    index=False
)


# ============================================================
# 9. Calculate mean and standard deviation
# ============================================================

summary = (
    fold_results
    .groupby("model")[
        [
            "macro_f1",
            "balanced_accuracy",
            "stress_recall"
        ]
    ]
    .agg(["mean", "std"])
)

summary.to_csv(
    OUTPUT_DIR / "person2_summary.csv"
)


# ============================================================
# 10. Print final summary
# ============================================================

print("\n\n====================================")
print("FINAL RESULTS")
print("====================================")

print(summary)