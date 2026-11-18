from pathlib import Path

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

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

SHAP_DIR = OUTPUT_DIR / "shap_results"
LIME_DIR = OUTPUT_DIR / "lime_results"

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
    raise ValueError(f"Missing columns: {missing_cols}")

X = df[FEATURE_COLS]
y = df["label"]
groups = df["subject_id"]

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Expected binary labels: 0 = Non-Stress, 1 = Stress"
    )


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
            class_weight="balanced",
            probability=True,
            random_state=42
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
# 6. SHAP Explainability
# ============================================================

def generate_shap_results(
    model,
    model_name,
    X_train,
    X_test,
    y_pred,
    held_out_subject
):

    print(f"\nGenerating SHAP for {model_name}...")

    # Background data from training subject groups only
    background = X_train.sample(
        n=min(25, len(X_train)),
        random_state=42
    )

    # Explain at most 20 held-out test windows
    X_explain = X_test.sample(
        n=min(20, len(X_test)),
        random_state=42
    )

    # -----------------------------------------
    # Create SHAP explainer
    # -----------------------------------------

    if model_name == "Random_Forest":

        explainer = shap.TreeExplainer(
            model,
            data=background,
            model_output="probability",
            feature_perturbation="interventional"
        )

        shap_output = explainer(X_explain)

    else:

        masker = shap.maskers.Independent(
            background.to_numpy(),
            max_samples=25
        )

        explainer = shap.Explainer(
            model.predict_proba,
            masker,
            algorithm="permutation",
            feature_names=FEATURE_COLS
        )

        shap_output = explainer(
            X_explain.to_numpy(),
            max_evals=2 * len(FEATURE_COLS) + 1
        )

    # -----------------------------------------
    # Extract Stress class SHAP values
    # -----------------------------------------

    classes = list(model.classes_)
    stress_class_idx = classes.index(1)

    values = np.asarray(shap_output.values)
    base_values = np.asarray(shap_output.base_values)

    if values.ndim == 3:
        stress_values = values[:, :, stress_class_idx]

    elif values.ndim == 2:
        stress_values = values

    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    if base_values.ndim == 2:
        stress_base_values = base_values[:, stress_class_idx]

    elif base_values.ndim == 1:
        stress_base_values = base_values

    else:
        raise ValueError(
            f"Unexpected SHAP base value shape: {base_values.shape}"
        )

    stress_explanation = shap.Explanation(
        values=stress_values,
        base_values=stress_base_values,
        data=X_explain.to_numpy(),
        feature_names=FEATURE_COLS
    )

    # -----------------------------------------
    # Output folder
    # -----------------------------------------

    subject_dir = (
        SHAP_DIR /
        str(held_out_subject) /
        model_name
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

    explained_predictions = pd.Series(
        y_pred,
        index=X_test.index
    ).loc[X_explain.index]

    stress_indices = np.where(
        explained_predictions.to_numpy() == 1
    )[0]

    # Prefer a window predicted as Stress
    sample_idx = (
        stress_indices[0]
        if len(stress_indices) > 0
        else 0
    )

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
    # D. Save numerical SHAP importance
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

    print(f"SHAP results saved: {subject_dir}")


# ============================================================
# 7. LIME Explainability
# ============================================================

def generate_lime_results(
    model,
    model_name,
    X_train,
    X_test,
    y_pred,
    held_out_subject
):

    print(f"\nGenerating LIME for {model_name}...")

    classes = list(model.classes_)
    stress_class_idx = classes.index(1)

    # -----------------------------------------
    # Select a test window
    # -----------------------------------------

    stress_indices = np.where(
        np.asarray(y_pred) == 1
    )[0]

    if len(stress_indices) > 0:

        sample_idx = stress_indices[0]

    else:

        sample_idx = 0

    sample = X_test.iloc[sample_idx]

    predicted_label = int(y_pred[sample_idx])

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
    # Output folder
    # -----------------------------------------

    subject_dir = (
        LIME_DIR /
        str(held_out_subject) /
        model_name
    )

    subject_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------
    # A. Save LIME explanation as HTML
    # -----------------------------------------

    explanation.save_to_file(
        str(subject_dir / "lime_explanation.html")
    )

    # -----------------------------------------
    # B. Save LIME explanation as PNG
    # -----------------------------------------

    fig = explanation.as_pyplot_figure(
        label=target_class_idx
    )

    fig.savefig(
        subject_dir / "lime_explanation.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close(fig)

    # -----------------------------------------
    # C. Save numerical LIME contributions
    # -----------------------------------------

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
        f"LIME results saved: {subject_dir} "
        f"| Explained prediction: {predicted_label}"
    )


# ============================================================
# 8. Run LOSO experiments
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

        # Fresh model for every fold
        model = build_model()

        # Train
        model.fit(
            X_train,
            y_train
        )

        # Predict
        y_pred = model.predict(X_test)

        # Held-out subject
        held_out_subject = (
            groups.iloc[test_idx].iloc[0]
        )

        # -----------------------------------------
        # SHAP
        # -----------------------------------------

        generate_shap_results(
            model=model,
            model_name=model_name,
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
            model_name=model_name,
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

    # -----------------------------------------
    # Pooled confusion matrix
    # -----------------------------------------

    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=[0, 1]
    )

    print("\nPooled confusion matrix:")
    print(cm)


# ============================================================
# 9. Save fold-level results
# ============================================================

fold_results = pd.DataFrame(
    all_fold_results
)

fold_results.to_csv(
    OUTPUT_DIR / "person2_fold_results.csv",
    index=False
)


# ============================================================
# 10. Save predictions
# ============================================================

predictions = pd.DataFrame(
    all_predictions
)

predictions.to_csv(
    OUTPUT_DIR / "person2_predictions.csv",
    index=False
)


# ============================================================
# 11. Calculate mean and standard deviation
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
# 12. Print final summary
# ============================================================

print("\n\n====================================")
print("FINAL RESULTS")
print("====================================")

print(summary)

print("\nAll model, SHAP and LIME results generated!")