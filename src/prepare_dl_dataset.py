
import pickle
import numpy as np
from pathlib import Path
from scipy.signal import resample_poly
from scipy.stats import mode

DATA_DIR = Path("data/WESAD")
OUTPUT_DIR = Path("outputs")

WINDOW_SIZE = 300
SUBJECTS = [
    "S2", "S3", "S4", "S5", "S6",
    "S7", "S8", "S9", "S10", "S11",
    "S13", "S14", "S15", "S16", "S17"
]

all_X = []
all_y = []
all_subjects = []

for subject in SUBJECTS:

    print(f"\nProcessing {subject}...")

    file_path = DATA_DIR / subject / f"{subject}.pkl"

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    chest = data["signal"]["chest"]

    eda = np.asarray(chest["EDA"]).flatten()
    temp = np.asarray(chest["Temp"]).flatten()
    labels = np.asarray(data["label"]).flatten()

    # Downsample signals from 700 Hz to 10 Hz
    eda_ds = resample_poly(eda, 1, 70)
    temp_ds = resample_poly(temp, 1, 70)

    # Majority voting for labels
    usable_length = (len(labels) // 70) * 70
    label_blocks = labels[:usable_length].reshape(-1, 70)

    labels_ds = mode(
        label_blocks,
        axis=1,
        keepdims=False
    ).mode

    # Check synchronization
    common_length = min(
        len(eda_ds), len(temp_ds), len(labels_ds)
    )

    eda_ds = eda_ds[:common_length]
    temp_ds = temp_ds[:common_length]
    labels_ds = labels_ds[:common_length]

    subject_X = []
    subject_y = []

    for start in range(
        0, common_length - WINDOW_SIZE + 1, WINDOW_SIZE
    ):

        end = start + WINDOW_SIZE

        label_window = labels_ds[start:end]

        if np.all(label_window == 1):
            target = 0
        elif np.all(label_window == 2):
            target = 1
        else:
            continue

        window = np.column_stack([
            eda_ds[start:end],
            temp_ds[start:end]
        ])

        if not np.isfinite(window).all():
            continue

        subject_X.append(window)
        subject_y.append(target)

    subject_X = np.asarray(subject_X, dtype=np.float32)
    subject_y = np.asarray(subject_y, dtype=np.int64)

    all_X.extend(subject_X)
    all_y.extend(subject_y)
    all_subjects.extend([subject] * len(subject_y))

    print("Windows:", len(subject_y))
    print("Baseline:", np.sum(subject_y == 0))
    print("Stress:", np.sum(subject_y == 1))

# Combine all subjects
X = np.asarray(all_X, dtype=np.float32)
y = np.asarray(all_y, dtype=np.int64)
subjects = np.asarray(all_subjects)

OUTPUT_DIR.mkdir(exist_ok=True)

np.savez_compressed(
    OUTPUT_DIR / "wesad_dl_dataset.npz",
    X=X,
    y=y,
    subjects=subjects
)

print("\n===== FINAL DATASET =====")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Subjects:", np.unique(subjects))
print("Total windows:", len(y))
print("Baseline:", np.sum(y == 0))
print("Stress:", np.sum(y == 1))
