import torch
import torch.nn as nn
import torch.nn.functional as F

class ae_model(nn.Module):
    #[batch_size==4, channels==1, height==1200, width==1200]
    def __init__(self):
        super(ae_model, self).__init__()
        # Encoder part
        self.enc = nn.Sequential(
            #block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # [B, 64, 1200, 1200]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # [B, 64, 1200, 1200]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 600, 600]
            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B, 128, 600, 600]
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # [B, 128, 600, 600]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 300, 300]
            #block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # [B, 256, 300, 300]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 256, 150, 150]
            #block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # [B, 512, 150, 150]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 512, 75, 75]
            #block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # [B, 512, 75, 75]
            nn.ReLU(),
            nn.MaxPool2d(3),  # [B, 512, 25, 25]
                        
            nn.Flatten(),  # Flatten the output for the linear layers
            nn.Linear(512 * 25 * 25, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU()
        )
        # Decoder part
        self.dec = nn.Sequential(
             nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512 * 25 * 25),
            nn.ReLU(),
            nn.Unflatten(1, (512, 25, 25)),  # [B, 512, 25, 25]
            nn.Upsample(scale_factor=3),  # [B, 512, 75, 75]
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),  # [B, 512, 75, 75]
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # [B, 512, 150, 150]
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),  # [B, 256, 150, 150]
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # [B, 256, 300, 300]
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),  # [B, 128, 300, 300]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),  # [B, 128, 300, 300]
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # [B, 128, 600, 600]
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),  # [B, 64, 600, 600]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),  # [B, 64, 600, 600]
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # [B, 64, 1200, 1200]
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1),  # [B, 1, 1200, 1200]
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode



