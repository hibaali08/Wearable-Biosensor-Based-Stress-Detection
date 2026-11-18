
import torch
from cnn_model import CNN1D

# Create model
model = CNN1D(input_channels=2)

# Simulate 8 windows, each with 300 time steps and 2 channels
dummy_input = torch.randn(8, 300, 2)

# Forward pass
output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
print("Model parameters:", sum(p.numel() for p in model.parameters()))

print("\nCNN forward pass successful!")
