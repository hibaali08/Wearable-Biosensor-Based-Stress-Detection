
import torch
from gru_model import GRUModel

model = GRUModel(input_size=2)

dummy_input = torch.randn(8, 300, 2)

output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
print(
    "Total parameters:",
    sum(p.numel() for p in model.parameters())
)
print("GRU forward pass successful!")
