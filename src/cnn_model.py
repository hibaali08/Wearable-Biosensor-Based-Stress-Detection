
import torch
import torch.nn as nn


class CNN1D(nn.Module):

    def __init__(self, input_channels=2):

        super(CNN1D, self).__init__()

        self.features = nn.Sequential(

            # First convolution block
            nn.Conv1d(
                input_channels,
                32,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Second convolution block
            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Third convolution block
            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        # Input: (batch, time_steps, channels)
        x = x.transpose(1, 2)

        x = self.features(x)

        x = self.classifier(x)

        return x.squeeze(1)
