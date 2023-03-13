import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8]

# Equalized Learning Rate (Equalized Convolutional Layer)
class WSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size**2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


# Pixel Normalization
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

# Regular Convolution Block with Pixel Normalization
# 2 Convolution Layers (3 x 3) with LeakyReLU activation
# Only the Generator uses pixel normalization "use_pixelnorm"
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pixelnorm = PixelNorm()
        self.use_pixelnorm = use_pixelnorm

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pixelnorm(x) if self.use_pixelnorm else x
        x = self.leaky(self.conv2(x))
        x = self.pixelnorm(x) if self.use_pixelnorm else x
        return x


class Generator(nn.Module):
    def __init__(self, latent_size, in_channels, img_channels=3):
        super().__init__()

        # 4x4
        self.conv_block1 = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(
                latent_size, in_channels, kernel_size=4, stride=1, padding=0
            ),  # 512x1x1 -> 512x4x4 (both this and the pixel norm should be substituted by a weighted scale transpose)
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )
        self.to_rgb1 = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )

        self.conv_blocks = nn.ModuleList()
        self.rgb_layers = nn.ModuleList([self.to_rgb1])

        # 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        for layer in range(len(factors) - 1):
            # factors[i] -> factors[i+1]
            in_channel = int(in_channels * factors[layer])
            out_channel = int(in_channels * factors[layer + 1])
            self.conv_blocks.append(ConvBlock(in_channel, out_channel))
            self.rgb_layers.append(
                WSConv2d(out_channel, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, up, out):
        return torch.tanh((1 - alpha) * up + alpha * out)

    def forward(self, x, alpha, steps):
        out = self.conv_block1(x)

        if steps == 0:
            return self.to_rgb1(out)

        for step in range(steps):
            # upsample
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.conv_blocks[step](upscaled)

        up_route = self.rgb_layers[steps - 1](upscaled)
        out_route = self.rgb_layers[steps](out)

        # apply fade in layers
        return self.fade_in(alpha, up_route, out_route)


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.rgb_layers = nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        # 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        for layer in range(len(factors) - 1, 0, -1):
            # factors[i] -> factors[i-1]
            in_channel = int(in_channels * factors[layer])
            out_channel = int(in_channels * factors[layer - 1])
            self.conv_blocks.append(
                ConvBlock(in_channel, out_channel, use_pixelnorm=False)
            )
            self.rgb_layers.append(
                WSConv2d(img_channels, in_channel, kernel_size=1, stride=1, padding=0)
            )

        # 4x4
        self.to_rgb1 = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.to_rgb1)
        # downsampling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_block1 = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def fade_in(self, alpha, down, out):
        return (1 - alpha) * down + alpha * out

    def minibatch_std(self, x):
        batch_stats = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_stats], dim=1)

    def forward(self, x, alpha, steps):

        current_step = len(self.conv_blocks) - steps

        out = self.leaky(self.rgb_layers[current_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.conv_block1(out).view(out.shape[0], -1)

        down = self.leaky(self.rgb_layers[current_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.conv_blocks[current_step](out))
        out = self.fade_in(alpha, down, out)

        for step in range(current_step + 1, len(self.conv_blocks)):
            out = self.conv_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.conv_block1(out).view(out.shape[0], -1)


if __name__ == "__main__":
    latent_size = 512
    in_channels = 256
    image_channels = 3
    alpha = 0.5

    generator = Generator(latent_size, in_channels, image_channels)
    discriminator = Discriminator(in_channels, image_channels)

    for image_size in [4, 8, 16, 32, 64, 128, 256]:

        steps = int(log2(image_size / 4))
        x = torch.randn(1, latent_size, 1, 1)
        z = generator(x, alpha, steps)
        assert z.shape == (1, image_channels, image_size, image_size)

        real = discriminator(z, alpha, steps)
        assert real.shape == (1, 1)

        print(f"Image size: {image_size}")
