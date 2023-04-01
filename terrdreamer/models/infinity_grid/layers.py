import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation="elu",
        reflection_padding: bool = True,
        dilation: int = 1,
    ):
        super(ConvBlock, self).__init__()

        if reflection_padding:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, dilation=dilation
                ),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
            )

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.conv(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class EncoderDecoderNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super(EncoderDecoderNetwork, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, 5, 1, 2, activation="elu"),
            ConvBlock(32, 64, 3, 2, 1, activation="elu"),
            ConvBlock(64, 64, 3, 1, 1, activation="elu"),
            ConvBlock(64, 128, 3, 2, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 2, dilation=2, activation="elu"),
            ConvBlock(128, 128, 3, 1, 4, dilation=4, activation="elu"),
            ConvBlock(128, 128, 3, 1, 8, dilation=8, activation="elu"),
            ConvBlock(128, 128, 3, 1, 16, dilation=16, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(128, 64, 3, 1, 1, activation="elu"),
            ConvBlock(64, 64, 3, 1, 1, activation="elu"),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(64, 32, 3, 1, 1, activation="elu"),
            ConvBlock(32, 16, 3, 1, 1, activation="elu"),
            ConvBlock(16, out_channels, 3, 1, 1, activation=None),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.clip(x, -1, 1)


if __name__ == "__main__":
    import torch

    x = torch.rand(8, 3, 256, 256)
    B, _, H, W = x.shape
    x_ones = torch.ones(B, 1, H, W)
    # Replace by a realistic mask - should be 1 for masked pixels and 0 for unmasked pixels
    mask = torch.ones(B, 1, H, W)
    x = x * (1 - mask)
    x = torch.cat([x, x_ones, x_ones * mask], dim=1)
    enc = EncoderDecoderNetwork(in_channels=3 + 1 + 1)
    y = enc(x)

    import pdb

    pdb.set_trace()
