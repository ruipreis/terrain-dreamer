import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(m, mean, std):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class UpSampleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
        dropout=False,
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.deconv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.upsample(x)
        x = self.deconv(x)

        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)

        if self.activation:
            x = self.act(x)

        return x


class DownSampleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
    ):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, d: int = 64):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, d, batchnorm=False),  # bs x 64 x 128 x 128
            DownSampleConv(d, d * 2),  # bs x d*2 x 64 x 64
            DownSampleConv(d * 2, d * 4),  # bs x d*4 x 32 x 32
            DownSampleConv(d * 4, d * 8),  # bs x d*8 x 16 x 16
            DownSampleConv(d * 8, d * 8),  # bs x d*8 x 8 x 8
            DownSampleConv(d * 8, d * 8),  # bs x d*8 x 4 x 4
            DownSampleConv(d * 8, d * 8),  # bs x d*8 x 2 x 2
            DownSampleConv(d * 8, d * 8, batchnorm=False),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(d * 8, d * 8, dropout=True),  # bs x d*8 x 2 x 2
            UpSampleConv(d * 8 * 2, d * 8, dropout=True),  # bs x d*8 x 4 x 4
            UpSampleConv(d * 8 * 2, d * 8, dropout=True),  # bs x d*8 x 8 x 8
            UpSampleConv(d * 8 * 2, d * 8),  # bs x d*8 x 16 x 16
            UpSampleConv(d * 8 * 2, d * 4),  # bs x d*4 x 32 x 32
            UpSampleConv(d * 8, d * 2),  # bs x 128 x 64 x 64
            UpSampleConv(d * 4, d),  # bs x 64 x 128 x 128
        ]
        self.final_conv = UpSampleConv(
            d, out_channels, activation=False, batchnorm=False
        )
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        x = self.final_conv(x)
        return self.tanh(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad


class BasicDiscriminator(nn.Module):
    def __init__(self, input_channels, d: int = 64):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, d, batchnorm=False)
        self.d2 = DownSampleConv(d, d * 2)
        self.d3 = DownSampleConv(d * 2, d * 4)
        self.d4 = DownSampleConv(d * 4, d * 8)
        self.final = nn.Conv2d(d * 8, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad


if __name__ == "__main__":
    input_rgb = torch.randn(8, 3, 256, 256)
    target_dem = torch.randn(8, 1, 256, 256)

    disc = BasicDiscriminator(3 + 1)
    print("Experimenting with the discriminator...")
    print(disc(input_rgb, target_dem).shape)

    generator = UNetGenerator(3, 1)
    print("Experimenting with the generator...")
    print(generator(input_rgb).shape)
