
import numpy as np

data = np.load("outputs/wesad_dl_dataset.npz")

X = data["X"]
y = data["y"]
subjects = data["subjects"]

unique_subjects = sorted(
    np.unique(subjects),
    key=lambda s: int(s[1:])
)

print("\nCLASS-WISE SIGNAL ANALYSIS")
print("=" * 90)

header = (
    "{:<8} {:<12} {:<12} {:<12} {:<12} {:<12}"
).format(
    "Subject",
    "Class",
    "EDA Mean",
    "EDA Std",
    "Temp Mean",
    "Temp Std"
)

print(header)
print("-" * 90)

for subject in unique_subjects:

    for label in [0, 1]:

        mask = (subjects == subject) & (y == label)

        if np.sum(mask) == 0:
            continue

        signal = X[mask]

        eda = signal[:, :, 0]
        temp = signal[:, :, 1]

        class_name = "Baseline" if label == 0 else "Stress"

        print(
            "{:<8} {:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                subject,
                class_name,
                eda.mean(),
                eda.std(),
                temp.mean(),
                temp.std()
            )
        )

print("\nANALYSIS COMPLETE")

