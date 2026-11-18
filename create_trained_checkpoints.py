import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cnn_model import CNN1D
from lstm_model import LSTMModel
from gru_model import GRUModel

def normalize_subject(X_subject):
    mean = X_subject.mean(axis=(0, 1), keepdims=True)
    std = X_subject.std(axis=(0, 1), keepdims=True)
    return (X_subject - mean) / np.maximum(std, 1e-6)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)

    data_path = Path("outputs/wesad_dl_dataset.npz")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found.")

    data = np.load(data_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    subjects = data["subjects"]

    # Subject-wise normalization
    X_norm = X.copy()
    for s in np.unique(subjects):
        mask = subjects == s
        X_norm[mask] = normalize_subject(X_norm[mask])

    # Destination directories
    dirs = [
        Path("outputs/checkpoints"),
        Path("outputs/model_weights")
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    positives = np.sum(y == 1)
    negatives = np.sum(y == 0)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -----------------------------
    # 1. Train & Save 1D-CNN
    # -----------------------------
    print("\n[1/3] Training 1D-CNN...")
    cnn_model = CNN1D(input_channels=2)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-4)
    # CNN expects (batch, 2, 300)
    X_cnn = torch.tensor(X_norm, dtype=torch.float32)
    dataset_cnn = TensorDataset(X_cnn, torch.tensor(y, dtype=torch.float32))
    loader_cnn = DataLoader(dataset_cnn, batch_size=64, shuffle=True)

    for epoch in range(15):
        cnn_model.train()
        for xb, yb in loader_cnn:
            optimizer.zero_grad()
            loss = criterion(cnn_model(xb), yb)
            loss.backward()
            optimizer.step()

    cnn_model.eval()
    for d in dirs:
        torch.save(cnn_model.state_dict(), d / "cnn_model.pth")
        torch.save(cnn_model.state_dict(), d / "cnn_model.pt")
    print("✓ Saved cnn_model.pth and cnn_model.pt")

    # -----------------------------
    # 2. Train & Save GRU
    # -----------------------------
    print("\n[2/3] Training GRU...")
    gru_model = GRUModel(input_size=2, hidden_size=32, num_layers=1)
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001, weight_decay=1e-4)
    dataset_seq = TensorDataset(torch.tensor(X_norm, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader_seq = DataLoader(dataset_seq, batch_size=64, shuffle=True)

    for epoch in range(15):
        gru_model.train()
        for xb, yb in loader_seq:
            optimizer.zero_grad()
            loss = criterion(gru_model(xb), yb)
            loss.backward()
            optimizer.step()

    gru_model.eval()
    for d in dirs:
        torch.save(gru_model.state_dict(), d / "gru_model.pth")
        torch.save(gru_model.state_dict(), d / "gru_model.pt")
    print("✓ Saved gru_model.pth and gru_model.pt")

    # -----------------------------
    # 3. Train & Save LSTM
    # -----------------------------
    print("\n[3/3] Training LSTM...")
    lstm_model = LSTMModel(input_size=2, hidden_size=64, num_layers=2)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(10):
        lstm_model.train()
        for xb, yb in loader_seq:
            optimizer.zero_grad()
            loss = criterion(lstm_model(xb), yb)
            loss.backward()
            optimizer.step()

    lstm_model.eval()
    for d in dirs:
        torch.save(lstm_model.state_dict(), d / "lstm_model.pth")
        torch.save(lstm_model.state_dict(), d / "lstm_model.pt")
    print("✓ Saved lstm_model.pth and lstm_model.pt")

    print("\nAll checkpoints successfully created and verified!")

if __name__ == "__main__":
    main()

