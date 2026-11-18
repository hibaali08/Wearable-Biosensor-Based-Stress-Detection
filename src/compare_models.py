
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------
# 1. CONFIGURATION
# --------------------------------

BASE_DIR = Path("outputs")

MODEL_FILES = {
    "CNN": BASE_DIR / "cnn_loso_subjectwise" / "cnn_fold_results.csv",
    "LSTM": BASE_DIR / "lstm_loso_subjectwise" / "lstm_fold_results.csv",
    "GRU": BASE_DIR / "gru_loso_subjectwise" / "gru_fold_results.csv"
}

OUTPUT_DIR = BASE_DIR / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "macro_f1",
    "balanced_accuracy",
    "stress_recall"
]


# --------------------------------
# 2. LOAD AND VALIDATE RESULTS
# --------------------------------

model_data = {}

for model_name, file_path in MODEL_FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing results file: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = ["subject_id"] + METRICS

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} is missing columns: {missing}"
        )

    if df["subject_id"].duplicated().any():
        raise ValueError(
            f"Duplicate subjects found in {model_name}"
        )

    if len(df) != 15:
        raise ValueError(
            f"{model_name} has {len(df)} folds instead of 15."
        )

    model_data[model_name] = df

    print(f"{model_name}: {len(df)} folds loaded.")


# --------------------------------
# 3. CHECK SUBJECT CONSISTENCY
# --------------------------------

reference_subjects = set(
    model_data["CNN"]["subject_id"]
)

for model_name, df in model_data.items():

    if set(df["subject_id"]) != reference_subjects:
        raise ValueError(
            f"Subject mismatch in {model_name}"
        )

print("\nAll three models contain the same 15 subjects.")


# --------------------------------
# 4. GENERATE SUMMARY TABLE
# --------------------------------

summary_rows = []

for model_name, df in model_data.items():

    for metric in METRICS:

        summary_rows.append({
            "Model": model_name,
            "Metric": metric,
            "Mean": df[metric].mean(),
            "Standard Deviation": df[metric].std()
        })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "model_comparison_summary.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(summary_df.to_string(index=False))


# --------------------------------
# 5. PER-SUBJECT COMPARISON
# --------------------------------

for metric in METRICS:

    comparison = None

    for model_name, df in model_data.items():

        temp = df[["subject_id", metric]].rename(
            columns={metric: model_name}
        )

        if comparison is None:
            comparison = temp
        else:
            comparison = comparison.merge(
                temp,
                on="subject_id",
                validate="one_to_one"
            )

    comparison.to_csv(
        OUTPUT_DIR / f"per_subject_{metric}.csv",
        index=False
    )


# --------------------------------
# 6. MACRO-F1 GRAPH
# --------------------------------

models = list(model_data.keys())

means = [
    model_data[m]["macro_f1"].mean() * 100
    for m in models
]

stds = [
    model_data[m]["macro_f1"].std() * 100
    for m in models
]

plt.figure(figsize=(8, 5))

plt.bar(
    models,
    means,
    yerr=stds,
    capsize=6
)

plt.ylabel("Macro-F1 (%)")
plt.xlabel("Deep Learning Architecture")
plt.title("Macro-F1 Comparison Across 15 LOSO Folds")
plt.ylim(0, 110)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "macro_f1_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------
# 7. BALANCED ACCURACY AND RECALL
# --------------------------------

x = np.arange(len(models))
width = 0.35

balanced_means = [
    model_data[m]["balanced_accuracy"].mean() * 100
    for m in models
]

recall_means = [
    model_data[m]["stress_recall"].mean() * 100
    for m in models
]

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    balanced_means,
    width,
    label="Balanced Accuracy"
)

plt.bar(
    x + width / 2,
    recall_means,
    width,
    label="Stress Recall"
)

plt.xticks(x, models)
plt.ylabel("Score (%)")
plt.title("Balanced Accuracy and Stress Recall Comparison")
plt.ylim(0, 110)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "accuracy_recall_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------
# 8. COMPLETION
# --------------------------------

print("\nComparison completed successfully!")
print("Results saved in:", OUTPUT_DIR)
