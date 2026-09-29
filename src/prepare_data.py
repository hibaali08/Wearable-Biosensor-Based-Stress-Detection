import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/WESAD")
OUT_FILE = Path("outputs/wesad_features.csv")

SUBJECTS = [
    "S2", "S3", "S4", "S5", "S6",
    "S7", "S8", "S9", "S10", "S11",
    "S13", "S14", "S15", "S16", "S17"
]

FS = 700
WINDOW_SECONDS = 30
WINDOW_SIZE = FS * WINDOW_SECONDS

def extract_features(signal):
    signal = np.asarray(signal, dtype=float).flatten()
    signal = signal[np.isfinite(signal)]

    if len(signal) == 0:
        return None

    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal)
    ]

rows = []

for subject in SUBJECTS:
    file_path = DATA_DIR / subject / f"{subject}.pkl"

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    chest = data["signal"]["chest"]
    eda = np.asarray(chest["EDA"]).flatten()
    temp = np.asarray(chest["Temp"]).flatten()
    labels = np.asarray(data["label"]).flatten()

    if not (len(eda) == len(temp) == len(labels)):
        raise ValueError(f"Signal/label length mismatch for {subject}")

    window_id = 0

    for start in range(0, len(labels) - WINDOW_SIZE + 1, WINDOW_SIZE):
        end = start + WINDOW_SIZE
        label_window = labels[start:end]

        # Keep only windows fully labelled as baseline or stress
        if np.all(label_window == 1):
            y = 0  # baseline
        elif np.all(label_window == 2):
            y = 1  # stress
        else:
            continue

        eda_features = extract_features(eda[start:end])
        temp_features = extract_features(temp[start:end])

        if eda_features is None or temp_features is None:
            continue

        rows.append({
            "subject_id": subject,
            "window_id": window_id,
            "label": y,
            "eda_mean": eda_features[0],
            "eda_std": eda_features[1],
            "eda_min": eda_features[2],
            "eda_max": eda_features[3],
            "temp_mean": temp_features[0],
            "temp_std": temp_features[1],
            "temp_min": temp_features[2],
            "temp_max": temp_features[3],
        })

        window_id += 1

df = pd.DataFrame(rows)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)
print("Shape:", df.shape)
print(df.groupby(["subject_id", "label"]).size())