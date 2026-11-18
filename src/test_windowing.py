
import pickle
import numpy as np

from scipy.signal import resample_poly
from scipy.stats import mode

# Load S2
with open("data/WESAD/S2/S2.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

chest = data["signal"]["chest"]

eda = np.asarray(chest["EDA"]).flatten()
temp = np.asarray(chest["Temp"]).flatten()
labels = np.asarray(data["label"]).flatten()

# Downsample from 700 Hz to 10 Hz
eda_ds = resample_poly(eda, 1, 70)
temp_ds = resample_poly(temp, 1, 70)

# Downsample labels using majority voting
usable_length = (len(labels) // 70) * 70
label_blocks = labels[:usable_length].reshape(-1, 70)

labels_ds = mode(
    label_blocks, axis=1, keepdims=False
).mode

# Create 30-second windows
window_size = 300

X = []
y = []

for start in range(0, len(labels_ds) - window_size + 1, window_size):

    end = start + window_size

    eda_window = eda_ds[start:end]
    temp_window = temp_ds[start:end]
    label_window = labels_ds[start:end]

    # Keep only completely baseline or stress windows
    if np.all(label_window == 1):
        target = 0

    elif np.all(label_window == 2):
        target = 1

    else:
        continue

    # Combine EDA and temperature
    window = np.column_stack([
        eda_window,
        temp_window
    ])

    X.append(window)
    y.append(target)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

# Save S2 test windows
np.savez_compressed(
    "outputs/s2_windows.npz",
    X=X,
    y=y,
    subject_id=np.array(["S2"] * len(y))
)

print("WINDOWING COMPLETE")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nBaseline windows:", np.sum(y == 0))
print("Stress windows:", np.sum(y == 1))
print("Total windows:", len(y))
