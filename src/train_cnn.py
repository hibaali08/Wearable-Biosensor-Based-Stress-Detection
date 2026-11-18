
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix
)

import matplotlib.pyplot as plt

from cnn_model import CNN1D


# -----------------------------
# 1. CONFIGURATION
# -----------------------------

SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 12

DATA_PATH = Path("outputs/wesad_dl_dataset.npz")
OUTPUT_DIR = Path("outputs/cnn_loso")

DEVICE = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# 2. LOAD DATA
# -----------------------------

data = np.load(DATA_PATH)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
subjects = data["subjects"]

unique_subjects = sorted(
    np.unique(subjects),
    key=lambda s: int(s[1:])
)

print("Dataset shape:", X.shape)
print("Total subjects:", len(unique_subjects))
print("Device:", DEVICE)


# -----------------------------
# 3. TRAIN ONE LOSO FOLD
# -----------------------------

def train_fold(test_subject, fold_number):

    print(f"\n========== FOLD {fold_number} ==========")
    print("Test subject:", test_subject)

    set_seed(SEED + fold_number)

    # Separate test participant
    test_mask = subjects == test_subject
    train_pool_mask = ~test_mask

    X_pool = X[train_pool_mask]
    y_pool = y[train_pool_mask]
    groups_pool = subjects[train_pool_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # Subject-independent validation split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED + fold_number
    )

    train_idx, val_idx = next(
        splitter.split(
            X_pool,
            y_pool,
            groups=groups_pool
        )
    )

    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    X_val = X_pool[val_idx]
    y_val = y_pool[val_idx]

    train_subjects = np.unique(groups_pool[train_idx])
    val_subjects = np.unique(groups_pool[val_idx])

    print("Training subjects:", len(train_subjects))
    print("Validation subjects:", val_subjects)
    print("Testing subject:", test_subject)

    # Fit normalization on training data only
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train, dtype=torch.float32)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Build model
    model = CNN1D(input_channels=2).to(DEVICE)

    # Handle class imbalance using training data only
    positives = np.sum(y_train == 1)
    negatives = np.sum(y_train == 0)

    pos_weight = torch.tensor(
        [negatives / max(positives, 1)],
        dtype=torch.float32
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    # -----------------------------
    # 4. TRAINING LOOP
    # -----------------------------

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(yb)

        train_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_running_loss += loss.item() * len(yb)

        val_loss = val_running_loss / len(val_dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        # Early stopping based on validation loss
        if val_loss < best_val_loss - 1e-5:

            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping at epoch:", epoch + 1)
            break

    # Restore best model
    if best_state is None:
        raise RuntimeError("No valid model checkpoint was saved.")

    model.load_state_dict(best_state)

    # -----------------------------
    # 5. TEST ON UNSEEN SUBJECT
    # -----------------------------

    model.eval()

    with torch.no_grad():

        test_tensor = torch.tensor(X_test).to(DEVICE)
        logits = model(test_tensor)

        probabilities = torch.sigmoid(logits).cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)

    macro_f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_test, predictions
    )

    stress_recall = recall_score(
        y_test, predictions,
        pos_label=1,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    print("\nTEST RESULTS")
    print("Macro-F1:", round(macro_f1, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Stress Recall:", round(stress_recall, 4))
    print("Confusion Matrix:\n", cm)

    # Save loss curve
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"CNN Loss - {test_subject}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"loss_{test_subject}.png"
    )
    plt.close()

    # Save predictions
    prediction_df = pd.DataFrame({
        "subject_id": [test_subject] * len(y_test),
        "actual": y_test,
        "predicted": predictions,
        "stress_probability": probabilities
    })

    return {
        "subject_id": test_subject,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "stress_recall": stress_recall,
        "confusion_matrix": cm,
        "predictions": prediction_df
    }


# -----------------------------
# 6. RUN LOSO EXPERIMENT
# -----------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        default="all",
        help="Number of LOSO folds to run, or 'all'"
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.folds == "all":
        selected_subjects = unique_subjects
    else:
        selected_subjects = unique_subjects[
            :int(args.folds)
        ]

    results = []
    all_predictions = []
    total_cm = np.zeros((2, 2), dtype=int)

    for i, subject in enumerate(selected_subjects, start=1):

        result = train_fold(subject, i)

        results.append({
            "subject_id": result["subject_id"],
            "macro_f1": result["macro_f1"],
            "balanced_accuracy": result["balanced_accuracy"],
            "stress_recall": result["stress_recall"]
        })

        all_predictions.append(result["predictions"])
        total_cm += result["confusion_matrix"]

    # Save results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "cnn_fold_results.csv",
        index=False
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "cnn_predictions.csv",
        index=False
    )

    # Summary
    summary = results_df.drop(columns=["subject_id"]).agg(
        ["mean", "std"]
    )

    summary.to_csv(
        OUTPUT_DIR / "cnn_summary.csv"
    )

    print("\n========== CNN SUMMARY ==========")
    print(summary)

    print("\nCombined confusion matrix:")
    print(total_cm)

    # Save combined confusion matrix
    plt.figure()
    plt.imshow(total_cm)
    plt.title("CNN - Combined LOSO Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], ["Baseline", "Stress"])
    plt.yticks([0, 1], ["Baseline", "Stress"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, total_cm[i, j],
                     ha="center", va="center")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "cnn_confusion_matrix.png"
    )
    plt.close()

    print("\nAll requested folds completed!")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
