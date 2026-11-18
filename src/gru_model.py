
import torch
import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(
        self,
        input_size=2,
        hidden_size=32,
        num_layers=1,
        dropout=0.3
    ):

        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        last_output = output[:, -1, :]

        logits = self.classifier(last_output)

        return logits.squeeze(1)
