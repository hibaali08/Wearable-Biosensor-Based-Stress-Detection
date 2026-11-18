# Wearable Biosensor-Based Stress Detection (WESAD)

This repository implements and evaluates machine learning and deep learning pipelines for binary stress detection using wearable physiological signals from the **WESAD** (Wearable Stress and Affect Detection) dataset.

---

## 📌 Problem Formulation
* **Task:** Binary Stress Classification
  * **Class 0:** Baseline (Non-Stress)
  * **Class 1:** Stress (Trier Social Stress Test condition)
* **Physiological Modalities (Chest Sensors):**
  * **EDA (Electrodermal Activity):** Sympathetic nervous system arousal
  * **TEMP (Skin Temperature):** Peripheral vasoconstriction response
* **Evaluation Protocol:** Strict **Leave-One-Subject-Out (LOSO)** cross-validation across all 15 subjects (`S2`–`S17`). For each fold, one complete subject is held out exclusively for testing to evaluate true subject-independent generalization.

---

## 🔬 Data Pipeline & Preprocessing

1. **Downsampling & Filtering:**
   * Raw chest signals recorded at 700 Hz are resampled to **10 Hz** (1:70 polyphase decimation) to preserve physiological dynamics while reducing computational footprint.
2. **Windowing:**
   * Signals are segmented into **30-second windows** (300 timesteps at 10 Hz) with majority voting applied to labels.
   * Dataset shape: `(889, 300, 2)` for 889 total windows across 15 subjects.
3. **Subject-Wise Normalization:**
   * Physiological biosignals exhibit large baseline variations between individuals.
   * Each participant's windows are normalized using their own signal statistics ($z$-score):
     $$\hat{X}_{\text{subject}} = \frac{X_{\text{subject}} - \mu_{\text{subject}}}{\sigma_{\text{subject}}}$$

---

## 📊 Benchmark Results (15-Fold LOSO Evaluation)

All deep learning architectures were evaluated on identical subject splits using Leave-One-Subject-Out cross-validation across all 15 WESAD subjects:

| Model Architecture | Macro-F1 (Mean ± Std) | Balanced Accuracy (Mean ± Std) | Stress Recall (Mean ± Std) | Key Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **1D-CNN** | 0.9207 ± 0.1004 | 0.9202 ± 0.1028 | 0.8913 ± 0.1597 | Fast local feature extraction across temporal filters |
| **LSTM** | 0.9250 ± 0.1237 | 0.9278 ± 0.1224 | 0.9191 ± 0.1641 | 2-layer recurrent memory with cell state tracking |
| **GRU** *(Top Performer)* | **0.9322 ± 0.1013** | **0.9369 ± 0.0974** | **0.9372 ± 0.1380** | **Highest accuracy & recall with minimal parameters (4.5k)** |

### 💡 Key Findings:
* **GRU achieved the best overall performance:** With single-layer gating and 32 hidden units, GRU avoids overfitting on small biosignal sample sizes (889 windows) while effectively modeling 30-second temporal dependencies.
* **Complementary Pair for Explainability (XAI):**
  * **1D-CNN + GRU** forms the optimal duo for Explainable AI (SHAP / Integrated Gradients / LIME):
    * **1D-CNN** explains local morphological features (sudden EDA peaks, pulse slopes).
    * **GRU** explains sustained temporal trends (temperature drops and prolonged sympathetic tone).

---

## 🧠 Pretrained Checkpoints & Explainable AI (XAI) Support

Trained PyTorch weights are provided in `outputs/checkpoints/` for quick inference, evaluation, and Explainable AI analysis:

* `outputs/checkpoints/cnn_model.pth` (1D-CNN)
* `outputs/checkpoints/gru_model.pth` (GRU)
* `outputs/checkpoints/lstm_model.pth` (LSTM)

### Loading Checkpoints in Python:
```python
import torch
from src.cnn_model import CNN1D
from src.gru_model import GRUModel
from src.lstm_model import LSTMModel

# 1D-CNN (Expects: batch, 2, 300)
cnn = CNN1D(input_channels=2)
cnn.load_state_dict(torch.load("outputs/checkpoints/cnn_model.pth"))
cnn.eval()

# GRU (Expects: batch, 300, 2)
gru = GRUModel(input_size=2, hidden_size=32, num_layers=1)
gru.load_state_dict(torch.load("outputs/checkpoints/gru_model.pth"))
gru.eval()

# LSTM (Expects: batch, 300, 2)
lstm = LSTMModel(input_size=2, hidden_size=64, num_layers=2)
lstm.load_state_dict(torch.load("outputs/checkpoints/lstm_model.pth"))
lstm.eval()
```

---

## 📁 Repository Structure

```text
Wearable-Biosensor-Based-Stress-Detection/
├── data/
│   └── WESAD/                        # Raw WESAD subject pickle files (S2 - S17)
├── outputs/
│   ├── wesad_features.csv            # Extracted hand-crafted features for classical ML
│   ├── wesad_dl_dataset.npz          # Processed 10 Hz windowed dataset (X, y, subjects)
│   ├── checkpoints/                  # Pretrained model weights (.pth and .pt)
│   ├── cnn_loso_subjectwise/         # 1D-CNN LOSO results, confusion matrices, loss curves
│   ├── lstm_loso_subjectwise/        # LSTM LOSO results, predictions, loss curves
│   ├── gru_loso_subjectwise/         # GRU LOSO results, predictions, loss curves
│   └── model_comparison/            # Aggregated comparison tables and benchmark plots
├── src/
│   ├── prepare_data.py               # Classical ML feature extraction
│   ├── prepare_dl_dataset.py         # Deep learning signal decimation & windowing
│   ├── run_logreg.py                 # Logistic Regression baseline
│   ├── person2_models.py             # SVM and Random Forest implementations
│   ├── person3_xgboost.py            # XGBoost implementation
│   ├── cnn_model.py                  # 1D-CNN PyTorch model definition
│   ├── train_cnn_subjectwise.py      # 1D-CNN LOSO training pipeline
│   ├── lstm_model.py                 # LSTM PyTorch model definition
│   ├── train_lstm.py                 # LSTM LOSO training pipeline
│   ├── gru_model.py                  # GRU PyTorch model definition
│   ├── train_gru.py                  # GRU LOSO training pipeline
│   ├── compare_models.py             # Automated benchmark comparison and plotting
│   └── inspect_wesad.py              # Signal inspection and diagnostic utilities
├── create_trained_checkpoints.py     # Fast generator for trained .pth checkpoints
└── README.MD
```

---

## 🚀 Getting Started

### 1. Environment Setup
Ensure Python 3.9+ and PyTorch are installed:
```bash
pip install torch numpy pandas scipy scikit-learn matplotlib
```

### 2. Data Preparation
To generate the windowed deep learning dataset (`outputs/wesad_dl_dataset.npz`) from raw WESAD files:
```bash
python src/prepare_dl_dataset.py
```

### 3. Training & Evaluating Models (LOSO Cross-Validation)

* **1D-CNN:**
  ```bash
  python src/train_cnn_subjectwise.py
  ```
* **LSTM:**
  ```bash
  python src/train_lstm.py
  ```
* **GRU:**
  ```bash
  python src/train_gru.py
  ```

*(To run fewer folds for quick testing, pass `--folds <number>`, e.g. `--folds 1`).*

### 4. Comparing All Models
To aggregate fold results from CNN, LSTM, and GRU and generate comparative visualizations:
```bash
python src/compare_models.py
```
This produces:
* `outputs/model_comparison/model_comparison_summary.csv`
* `outputs/model_comparison/macro_f1_comparison.png`
* `outputs/model_comparison/accuracy_recall_comparison.png`

### 5. Regenerating Pretrained Checkpoints
To generate or refresh standalone `.pth` checkpoints for Explainable AI:
```bash
python create_trained_checkpoints.py
```
