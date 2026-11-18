from pathlib import Path

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

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

SHAP_DIR = OUTPUT_DIR / "xgboost_shap_results"
LIME_DIR = OUTPUT_DIR / "xgboost_lime_results"

SHAP_DIR.mkdir(parents=True, exist_ok=True)
LIME_DIR.mkdir(parents=True, exist_ok=True)


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
        "Expected binary labels: 0 = Non-Stress, 1 = Stress"
    )

print("Total subjects:", groups.nunique())
print("Total windows:", len(df))
print("Class distribution:")
print(y.value_counts())


# ============================================================
# 4. LOSO setup
# ============================================================

logo = LeaveOneGroupOut()

fold_results = []
all_predictions = []

y_true_all = []
y_pred_all = []


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

    print("\nGenerating SHAP for XGBoost...")

    # Training data used as background
    background = X_train.sample(
        n=min(25, len(X_train)),
        random_state=42
    )

    # Explain maximum 20 test windows
    X_explain = X_test.sample(
        n=min(20, len(X_test)),
        random_state=42
    )

    # Create TreeExplainer
    explainer = shap.TreeExplainer(
        model,
        data=background,
        model_output="probability",
        feature_perturbation="interventional"
    )

    # Calculate SHAP values
    shap_output = explainer(X_explain)

    values = np.asarray(shap_output.values)
    base_values = np.asarray(shap_output.base_values)

    # Select Stress class (label 1)
    if values.ndim == 3:

        stress_class_idx = list(model.classes_).index(1)

        stress_values = values[:, :, stress_class_idx]

        if base_values.ndim == 2:
            stress_base_values = base_values[:, stress_class_idx]
        else:
            stress_base_values = base_values

    elif values.ndim == 2:

        # Binary XGBoost output generally has
        # one SHAP value per feature for positive class
        stress_values = values

        stress_base_values = base_values.reshape(-1)

    else:

        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    # Create explanation object
    stress_explanation = shap.Explanation(
        values=stress_values,
        base_values=stress_base_values,
        data=X_explain.to_numpy(),
        feature_names=FEATURE_COLS
    )

    # Subject-wise output folder
    subject_dir = (
        SHAP_DIR / str(held_out_subject)
    )

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
    # D. Save SHAP feature importance
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

    print("SHAP saved for subject:", held_out_subject)


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

    print("\nGenerating LIME for XGBoost...")

    # Select one test window predicted as Stress
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
    # Subject-wise output folder
    # -----------------------------------------

    subject_dir = (
        LIME_DIR / str(held_out_subject)
    )

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

    # Save numerical LIME contributions
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
        "LIME saved for subject:",
        held_out_subject,
        "| Explained class:",
        predicted_label
    )


# ============================================================
# 7. Run XGBoost with LOSO
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

    print("\n====================================")
    print("Fold:", fold_num)
    print("Held-out subject:", held_out_subject)
    print("====================================")

    # -----------------------------------------
    # Build fresh XGBoost model
    # -----------------------------------------

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict
    y_pred = model.predict(X_test)

    # -----------------------------------------
    # SHAP
    # -----------------------------------------

    generate_shap_results(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_pred=y_pred,
        held_out_subject=held_out_subject
    )

    # -----------------------------------------
    # LIME
    # -----------------------------------------

    generate_lime_results(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_pred=y_pred,
        held_out_subject=held_out_subject
    )

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

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
# 8. Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true_all,
    y_pred_all,
    labels=[0, 1]
)

print("\nPooled confusion matrix:")
print(cm)


# ============================================================
# 9. Save fold results
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)

fold_results_df.to_csv(
    OUTPUT_DIR / "person3_xgboost_fold_results.csv",
    index=False
)


# ============================================================
# 10. Save predictions
# ============================================================

predictions_df = pd.DataFrame(
    all_predictions
)

predictions_df.to_csv(
    OUTPUT_DIR / "person3_xgboost_predictions.csv",
    index=False
)


# ============================================================
# 11. Summary statistics
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
# 12. Print final result
# ============================================================

print("\n====================================")
print("XGBOOST FINAL RESULTS")
print("====================================")

print(summary)

print("\nAll XGBoost, SHAP and LIME results generated!")