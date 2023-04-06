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
        f_height = height // (2**4)
        f_width = width // (2**4)

        self.seq = nn.Sequential(
            ConvBlock(3, cnum),
            ConvBlock(cnum, cnum * 2),
            ConvBlock(cnum * 2, cnum * 4),
            ConvBlock(cnum * 4, cnum * 8),
            nn.Flatten(),
            nn.Linear(cnum * 8 * f_height * f_width, 1),
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
        )

        self.flatten = nn.Flatten()

        f_height = height // (2**4)
        f_width = width // (2**4)
        self.linear = nn.Linear(cnum * 4 * f_height * f_width, 1)

    def forward(self, x):
        x = self.seq(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def set_requires_grad(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad = requires_grad


class GeneralCritic(nn.Module):
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.local_critic = LocalCritic(height, width)
        self.global_critic = GlobalCritic(height, width)

    def forward(self, local_batch, global_batch):
        return self.local_critic(local_batch), self.global_critic(global_batch)

    def set_requires_grad(self, requires_grad: bool):
        self.local_critic.set_requires_grad(requires_grad)
        self.global_critic.set_requires_grad(requires_grad)
