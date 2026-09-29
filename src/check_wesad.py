import pickle
import numpy as np

# Load subject S2
path = "data/WESAD/S2/S2.pkl"

with open(path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

# First, inspect the available keys
print("Top-level keys:", data.keys())
print("Signal keys:", data["signal"].keys())

# Extract chest signals
eda = np.asarray(data["signal"]["chest"]["EDA"]).flatten()
temp = np.asarray(data["signal"]["chest"]["Temp"]).flatten()
labels = np.asarray(data["label"]).flatten()

# Print their shapes
print("EDA:", eda.shape)
print("Temp:", temp.shape)
print("Labels:", labels.shape)
print("Unique labels:", np.unique(labels))