import torch
import torch.nn as nn
import torchvision.models as models

class cat_model(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(cat_model, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        #model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1024)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
       

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        return x
