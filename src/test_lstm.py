
import torch
from lstm_model import LSTMModel

model = LSTMModel(input_size=2)

# Batch of 8 windows, 300 time steps, 2 signals
dummy_input = torch.randn(8, 300, 2)

output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))
print("LSTM forward pass successful!")
