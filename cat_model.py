import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class cat_model(nn.Module):
    def __init__(self, in_channels=1, features=8, num_classes=2):
        super(cat_model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # [B, 32, 1200, 1200]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [B, 32, 600, 600]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 600, 600]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [B, 64, 300, 300]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B, 128, 300, 300]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [B, 128, 150, 150]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # [B, 256, 150, 150]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [B, 256, 75, 75]

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # [B, 512, 75, 75]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),  # [B, 512, 25, 25]
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 25 * 25, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1)  
            # For binary classification, you might want to use a sigmoid activation
        )

    def forward(self, x):
        x = self.cnn(x)  # [B, 512, 25, 25]
        x = x.view(x.size(0), -1)  # Flatten [B, 512*25*25]
        x = self.classifier(x)  # Classify [B, num_classes]
        return x






