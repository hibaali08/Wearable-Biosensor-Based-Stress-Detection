from pathlib import Path

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Paths
# ============================================================

DATA_FILE = "outputs/wesad_features.csv"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHAP_DIR = OUTPUT_DIR / "logreg_shap_results"
LIME_DIR = OUTPUT_DIR / "logreg_lime_results"

SHAP_DIR.mkdir(parents=True, exist_ok=True)
LIME_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Features
# ============================================================

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_min", "eda_max",
    "temp_mean", "temp_std", "temp_min", "temp_max"
]


# ============================================================
# 3. Load dataset
# ============================================================

df = pd.read_csv(DATA_FILE)

required_cols = FEATURE_COLS + ["label", "subject_id"]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

X = df[FEATURE_COLS]
y = df["label"]
groups = df["subject_id"]

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Expected labels: 0 = Non-Stress, 1 = Stress"
    )

print("Total subjects:", groups.nunique())
print("Total windows:", len(df))


# ============================================================
# 4. LOSO setup
# ============================================================

logo = LeaveOneGroupOut()

fold_rows = []
all_true = []
all_pred = []
all_predictions = []


# ============================================================
# 5. SHAP Explainability
# ============================================================

def generate_shap_results(
    model,
    X_train,
    X_test,
    y_pred,
    held_out_subject
):

    print("\nGenerating SHAP...")

    # Background data from training subjects only
    background = X_train.sample(
        n=min(25, len(X_train)),
        random_state=42
    )

    # Explain maximum 20 test windows
    X_explain = X_test.sample(
        n=min(20, len(X_test)),
        random_state=42
    )

    # Create independent feature masker
    masker = shap.maskers.Independent(
        background.to_numpy(),
        max_samples=25
    )

    # Permutation Explainer works with the complete pipeline
    explainer = shap.Explainer(
        model.predict_proba,
        masker,
        algorithm="permutation",
        feature_names=FEATURE_COLS,
        output_names=["Non-Stress", "Stress"]
    )

    shap_output = explainer(
        X_explain.to_numpy(),
        max_evals=2 * len(FEATURE_COLS) + 1
    )

    values = np.asarray(shap_output.values)
    base_values = np.asarray(shap_output.base_values)

    # Select Stress class (label 1)
    stress_class_idx = list(model.classes_).index(1)

    if values.ndim == 3:

        stress_values = values[:, :, stress_class_idx]

    elif values.ndim == 2:

        stress_values = values

    else:

        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    if base_values.ndim == 2:

        stress_base_values = base_values[:, stress_class_idx]

    elif base_values.ndim == 1:

        stress_base_values = base_values

    else:

        stress_base_values = np.repeat(
            base_values.item(),
            len(X_explain)
        )

    # Create Stress-class explanation
    stress_explanation = shap.Explanation(
        values=stress_values,
        base_values=stress_base_values,
        data=X_explain.to_numpy(),
        feature_names=FEATURE_COLS
    )

    # Subject-wise folder
    subject_dir = SHAP_DIR / str(held_out_subject)

    subject_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------
    # A. SHAP Bar Plot
    # -----------------------------------------

    shap.plots.bar(
        stress_explanation,
        show=False
    )

    plt.savefig(
        subject_dir / "shap_bar.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    # -----------------------------------------
    # B. SHAP Beeswarm Plot
    # -----------------------------------------

    shap.plots.beeswarm(
        stress_explanation,
        show=False
    )

    plt.savefig(
        subject_dir / "shap_beeswarm.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    # -----------------------------------------
    # C. SHAP Waterfall Plot
    # -----------------------------------------

    # Find a test window predicted as Stress
    explained_predictions = pd.Series(
        y_pred,
        index=X_test.index
    ).loc[X_explain.index]

    stress_indices = np.where(
        explained_predictions.to_numpy() == 1
    )[0]

    if len(stress_indices) > 0:
        sample_idx = stress_indices[0]
    else:
        sample_idx = 0

    shap.plots.waterfall(
        stress_explanation[sample_idx],
        show=False
    )

    plt.savefig(
        subject_dir / "shap_waterfall.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    # -----------------------------------------
    # D. Save feature importance
    # -----------------------------------------

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_absolute_shap": np.abs(
            stress_values
        ).mean(axis=0)
    })

    importance = importance.sort_values(
        by="mean_absolute_shap",
        ascending=False
    )

    importance.to_csv(
        subject_dir / "shap_feature_importance.csv",
        index=False
    )

    print("SHAP results saved:", subject_dir)


# ============================================================
# 6. LIME Explainability
# ============================================================

def generate_lime_results(
    model,
    X_train,
    X_test,
    y_pred,
    held_out_subject
):

    print("\nGenerating LIME...")

    # Select a test window predicted as Stress
    stress_indices = np.where(
        np.asarray(y_pred) == 1
    )[0]

    if len(stress_indices) > 0:
        sample_idx = stress_indices[0]
    else:
        sample_idx = 0

    sample = X_test.iloc[sample_idx]

    predicted_label = int(y_pred[sample_idx])

    classes = list(model.classes_)

    target_class_idx = classes.index(predicted_label)

    # -----------------------------------------
    # Create LIME explainer
    # -----------------------------------------

    lime_explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=FEATURE_COLS,
        class_names=[
            "Non-Stress" if c == 0 else "Stress"
            for c in classes
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    # -----------------------------------------
    # Generate explanation
    # -----------------------------------------

    explanation = lime_explainer.explain_instance(
        data_row=sample.to_numpy(),
        predict_fn=model.predict_proba,
        num_features=len(FEATURE_COLS),
        labels=(target_class_idx,)
    )

    # -----------------------------------------
    # Subject-wise folder
    # -----------------------------------------

    subject_dir = LIME_DIR / str(held_out_subject)

    subject_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save HTML explanation
    explanation.save_to_file(
        str(subject_dir / "lime_explanation.html")
    )

    # Save PNG explanation
    fig = explanation.as_pyplot_figure(
        label=target_class_idx
    )

    fig.savefig(
        subject_dir / "lime_explanation.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close(fig)

    # Save numerical contributions
    lime_values = explanation.as_list(
        label=target_class_idx
    )

    lime_df = pd.DataFrame(
        lime_values,
        columns=[
            "feature_condition",
            "lime_contribution"
        ]
    )

    lime_df.to_csv(
        subject_dir / "lime_feature_importance.csv",
        index=False
    )

    print(
        "LIME results saved:",
        subject_dir,
        "| Explained class:",
        predicted_label
    )


# ============================================================
# 7. Run Logistic Regression with LOSO
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    held_out_subject = groups.iloc[test_idx].iloc[0]

    print("\n====================================")
    print("Fold:", fold)
    print("Held-out subject:", held_out_subject)
    print("====================================")

    # -----------------------------------------
    # Build Logistic Regression pipeline
    # -----------------------------------------

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict
    pred = model.predict(X_test)

    # -----------------------------------------
    # SHAP
    # -----------------------------------------

    generate_shap_results(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_pred=pred,
        held_out_subject=held_out_subject
    )

    # -----------------------------------------
    # LIME
    # -----------------------------------------

    generate_lime_results(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_pred=pred,
        held_out_subject=held_out_subject
    )

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        pred
    )

    stress_recall = recall_score(
        y_test,
        pred,
        pos_label=1,
        zero_division=0
    )

    fold_rows.append({
        "model": "Logistic_Regression",
        "held_out_subject": held_out_subject,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "stress_recall": stress_recall
    })

    # Store predictions
    all_true.extend(y_test.tolist())
    all_pred.extend(pred.tolist())

    for true_label, predicted_label in zip(
        y_test,
        pred
    ):

        all_predictions.append({
            "model": "Logistic_Regression",
            "held_out_subject": held_out_subject,
            "true_label": int(true_label),
            "predicted_label": int(predicted_label)
        })

    print(
        f"Fold {fold:02d} | "
        f"Subject {held_out_subject} | "
        f"Macro-F1 = {macro_f1:.4f} | "
        f"Balanced Acc = {balanced_acc:.4f} | "
        f"Stress Recall = {stress_recall:.4f}"
    )


# ============================================================
# 8. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1]
)

print("\nPooled confusion matrix:")
print(cm)


# ============================================================
# 9. Save fold results
# ============================================================

results = pd.DataFrame(fold_rows)

results.to_csv(
    OUTPUT_DIR / "logreg_fold_results.csv",
    index=False
)


# ============================================================
# 10. Save predictions
# ============================================================

predictions = pd.DataFrame(all_predictions)

predictions.to_csv(
    OUTPUT_DIR / "logreg_predictions.csv",
    index=False
)


# ============================================================
# 11. Summary statistics
# ============================================================

summary = (
    results[
        [
            "macro_f1",
            "balanced_accuracy",
            "stress_recall"
        ]
    ]
    .agg(["mean", "std"])
)

summary.to_csv(
    OUTPUT_DIR / "logreg_summary.csv"
)


# ============================================================
# 12. Print final results
# ============================================================

print("\n====================================")
print("LOGISTIC REGRESSION FINAL RESULTS")
print("====================================")

print(results)

print("\nMean fold scores:")
print(summary)

print("\nAll Logistic Regression, SHAP and LIME results generated!")
