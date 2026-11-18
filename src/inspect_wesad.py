
import pickle
import numpy as np

# Load subject S2
file_path = "data/WESAD/S2/S2.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

# Inspect dataset structure
print("Top-level keys:", data.keys())

# Extract chest signals
chest = data["signal"]["chest"]

eda = np.asarray(chest["EDA"]).flatten()
temp = np.asarray(chest["Temp"]).flatten()
labels = np.asarray(data["label"]).flatten()

# Display information
print("\nEDA shape:", eda.shape)
print("Temperature shape:", temp.shape)
print("Labels shape:", labels.shape)

print("\nUnique labels:", np.unique(labels))
print("EDA sampling rate: 700 Hz")
print("Temperature sampling rate: 700 Hz")

print("\nFirst 10 EDA values:", eda[:10])
print("First 10 temperature values:", temp[:10])
print("First 10 labels:", labels[:10])

print("\nAll lengths equal:",
      len(eda) == len(temp) == len(labels))



# Count samples belonging to each label
unique_labels, counts = np.unique(labels, return_counts=True)

print("\nLABEL DISTRIBUTION")

for label, count in zip(unique_labels, counts):
    percentage = (count / len(labels)) * 100
    print(
        f"Label {label}: {count} samples "
        f"({percentage:.2f}%)"
    )
