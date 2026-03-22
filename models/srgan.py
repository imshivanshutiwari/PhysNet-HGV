"""
Super-Resolution Generative Adversarial Network (SRGAN).

Enhances low-resolution optical imagery of HGV flight to improve 
tracking precision and feature identification.
"""

import torch
import torch.nn as nn
from typing import Tuple

class ResBlock(nn.Module):
    """
    Residual Block with Batch Normalization for SRGAN Generator.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class SRGANGenerator(nn.Module):
    """
    Generator Architecture: 16 ResBlocks + 2 PixelShuffle(x4 upsampling).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_res_blocks: int = 16):
        super().__init__()
        
        # Initial Conv
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # ResBlocks
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(n_res_blocks)])
        
        # Intermediate Conv
        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Upsampling (x4)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        # Final Conv
        self.final = nn.Conv2d(64, out_channels, kernel_size=9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.initial(x)
        feat2 = self.res_blocks(feat1)
        feat3 = self.mid(feat2)
        feat4 = self.upsample(feat1 + feat3) # Residual connection
        return torch.tanh(self.final(feat4))

class SRGANDiscriminator(nn.Module):
    """
    Discriminator Architecture: 8 ConvBlocks (VGG-style).
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        def conv_block(in_f, out_f, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            conv_block(64, 64, stride=2),
            conv_block(64, 128),
            conv_block(128, 128, stride=2),
            conv_block(128, 256),
            conv_block(256, 256, stride=2),
            conv_block(256, 512),
            conv_block(512, 512, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(x))

if __name__ == "__main__":
    # Test SRGAN
    gen = SRGANGenerator()
    disc = SRGANDiscriminator()
    
    lr_img = torch.randn(1, 3, 32, 32)
    hr_img = gen(lr_img)
    validity = disc(hr_img)
    
    print(f"Generator Input (LR): {lr_img.shape}")
    print(f"Generator Output (SR): {hr_img.shape}") # Should be 128x128
    print(f"Discriminator Validity: {validity.item():.4f}")
