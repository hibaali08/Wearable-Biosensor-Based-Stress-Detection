
import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, input_size=2, hidden_size=64,
                 num_layers=2, dropout=0.3):

        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_output = output[:, -1, :]

        logits = self.classifier(last_output)

        return logits.squeeze(1)
