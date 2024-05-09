import torch
import torch.nn as nn
import torch.nn.functional as F


class cat_model(nn.Module):
    def __init__(self, numClasses=2):
        super(cat_model, self).__init__()

        self.cat_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),  # padding='same'
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # padding='same'
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.attention_weights = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.attention_output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, numClasses),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cat_model(x)
        attention_weights = self.attention_weights(x)
        attention_output = self.attention_output(attention_weights * x)
        return attention_output
