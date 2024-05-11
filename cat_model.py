import torch
import torch.nn as nn
import torch.nn.functional as F

class cat_model(nn.Module):
    #[batch_size==4, channels==1, height==1200, width==1200]
    def __init__(self, in_channels=1, features= 8, num_classes=2):
        super(cat_model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),#[B, 32, 1200, 1200]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),  # [B, 32, 300, 300]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),#[B, 64, 300, 300]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #[B, 64, 150, 150]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), #[B, 128, 150, 150]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #[B, 128, 75, 75]
        )
        self.attention_weights = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),  # [B, 1, 75, 75]
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*75*75, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x) # [B, 128, 75, 75]
        attention_weights = self.attention_weights(x)  # [B, 1, 75, 75]
        x = x * attention_weights  # # [B, 128, 75, 75]
        x = x.view(x.size(0), -1)  # Flatten [B, 128*75*75]
        x = self.classifier(x)  # Classify [B, num_classes]
        return x



