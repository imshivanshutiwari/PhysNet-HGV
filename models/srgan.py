import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual


class Generator(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU())

        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(16)])

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.block4 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=9, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.res_blocks(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        return out4


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels) if in_channels != 3 else nn.Identity(),
                nn.LeakyReLU(0.2),
            )

        self.features = nn.Sequential(
            conv_block(3, 64, 1),
            conv_block(64, 64, 2),
            conv_block(64, 128, 1),
            conv_block(128, 128, 2),
            conv_block(128, 256, 1),
            conv_block(256, 256, 2),
            conv_block(256, 512, 1),
            conv_block(512, 512, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out


class SRGANLoss(nn.Module):
    def __init__(self, lambda_pixel=1e-2):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, fake_img, real_img, fake_score):
        content_loss = self.mse(fake_img, real_img)
        adversarial_loss = self.bce(fake_score, torch.ones_like(fake_score))
        pixel_loss = self.mse(fake_img, real_img)

        total_loss = content_loss + adversarial_loss + self.lambda_pixel * pixel_loss
        return total_loss
