import torch
import torch.nn as nn
import torch.nn.functional as F

class ae_model(nn.Module):
    #[batch_size==4, channels==1, height==1200, width==1200]
    def __init__(self):
        super(ae_model, self).__init__()
        # Encoder part
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B, 16, 1200, 1200]
            nn.ReLU(),
            nn.MaxPool2d(4),  # [B, 16, 300, 300]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 300, 300]
            nn.ReLU(),
            nn.MaxPool2d(4),  # [B, 32, 75, 75]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 75, 75]
            nn.ReLU(),
            nn.MaxPool2d(3),  # [B, 64, 25, 25]
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.Flatten(),  # Flatten the output for the linear layers
            nn.Linear(128 * 25 * 25, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.ReLU(),
        )
        # Decoder part
        self.dec = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 25 * 25),
            nn.ReLU(),
            # Reshape to [B, 64, 25, 25] before using ConvTranspose
            nn.Unflatten(1, (128, 25, 25)),
            nn.ConvTranspose2d(128, 64, kernel_size=3,  padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3, padding=1, output_padding=2),  # [B, 32, 75, 75]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=4, padding=1, output_padding=3),  # [B, 16, 300, 300]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=4, padding=1, output_padding=3),  # [B, 1, 1200, 1200]
            nn.Sigmoid() 
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode



