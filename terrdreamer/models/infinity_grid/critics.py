import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 5, stride=2, padding=2)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leaky_relu(self.conv(x))


class LocalCritic(nn.Module):
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        cnum = 64

        self.seq = nn.Sequential(
            ConvBlock(3, cnum),
            ConvBlock(cnum, cnum * 2),
            ConvBlock(cnum * 2, cnum * 4),
            ConvBlock(cnum * 4, cnum * 8),
            nn.Flatten(),
            nn.Linear(cnum * 8 * height * width, 1),
        )

    def forward(self, x):
        return self.seq(x)

    def set_requires_grad(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad = requires_grad


class GlobalCritic(nn.Module):
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        cnum = 64

        self.seq = nn.Sequential(
            ConvBlock(3, cnum),
            ConvBlock(cnum, cnum * 2),
            ConvBlock(cnum * 2, cnum * 4),
            ConvBlock(cnum * 4, cnum * 4),
            nn.Flatten(),
            nn.Linear(cnum * 4 * height * width, 1),
        )

    def forward(self, x):
        return self.seq(x)
    
    def set_requires_grad(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad = requires_grad
