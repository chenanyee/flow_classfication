import torch
import torch.nn as nn
import torch.nn.functional as F


class cat_model(nn.Module):
    def __init__(self):
        super(cat_model, self).__init__()
        # Define the layers of the model
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.attention_fc = nn.Linear(in_features=128, out_features=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        # Input x has shape (batch_size, 3, height, width)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Attention mechanism
        attention_weights = F.sigmoid(self.attention_fc(x))
        attention_weights = attention_weights.view(-1, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention weights to CNN output
        x = x.view(x.size(0), -1)
        attention_output = torch.bmm(attention_weights, x.unsqueeze(2)).squeeze(2)

        # Fully connected layer
        x = self.flatten(attention_output)
        x = F.softmax(self.fc(x), dim=1)
        return x
