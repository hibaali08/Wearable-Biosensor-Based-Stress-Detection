
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

# Downsample signals from 700 Hz to 10 Hz
eda_ds = resample_poly(eda, up=1, down=70)
temp_ds = resample_poly(temp, up=1, down=70)

# Convert labels using majority voting in each
# corresponding block of 70 original samples
labels_blocks = labels[:len(labels) // 70 * 70].reshape(-1, 70)

labels_ds = mode(
    labels_blocks,
    axis=1,
    keepdims=False
).mode

# Check results
print("Original EDA:", len(eda))
print("Downsampled EDA:", len(eda_ds))

print("Original temperature:", len(temp))
print("Downsampled temperature:", len(temp_ds))

print("Downsampled labels:", len(labels_ds))

print("\nAll lengths equal:",
      len(eda_ds) == len(temp_ds) == len(labels_ds))

print("\nNew sampling frequency: 10 Hz")

