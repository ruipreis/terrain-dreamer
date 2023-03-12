import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class _DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._conv = _DoubleConvolution(in_channels, out_channels)
        self._pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self._conv(x)
        p = self._pool(x)
        return x, p


class _UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # The number of channels needs to be halved in order to cat the skip
        # connection
        self._up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self._conv = _DoubleConvolution(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self._up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self._conv(x)


class _OutputConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BasicDiscriminator(nn.Module):
    def __init__(self, dest_channels: int = 512):
        super().__init__()

        src_base_channels = 64
        src_base_log2 = int(torch.log2(torch.tensor(src_base_channels)))
        dst_log2 = int(torch.log2(torch.tensor(dest_channels)))

        # This models increases the number of channels from 1 to 512,
        # the number of channels in doubled in each block, starting from 64
        # depending on the number of destination channels.
        layers = [_DoubleConvolution(1, src_base_channels)]

        for i in range(src_base_log2, dst_log2):
            in_channels = 2**i
            out_channels = 2 ** (i + 1)

            layers.append(_DownSample(in_channels, out_channels))

        # Add the final layer that applies a convolution with a kernel size of 1
        # to reduce the number of channels to 1
        layers.append(_OutputConvolution(dest_channels, 1))

        layers.append(nn.Sigmoid())

        self._sequence = nn.Sequential(*layers)
        self._layers = layers

    def forward(self, x):
        for layer in self._layers:
            if isinstance(layer, _DownSample):
                _,x = layer(x)
            else:
                x = layer(x)

        # # Average the output
        # x = torch.mean(x, dim=(2, 3))

        return x


class UNetGenerator(nn.Module):
    # Recieves an input shaped (B, 3, 256, 256)
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = _DownSample(3, 64)
        self.e2 = _DownSample(64, 128)
        self.e3 = _DownSample(128, 256)
        self.e4 = _DownSample(256, 512)

        """ Bottleneck """
        self.b = _DoubleConvolution(512, 1024)

        """ Decoder """
        self.d1 = _UpSample(1024, 512)
        self.d2 = _UpSample(512, 256)
        self.d3 = _UpSample(256, 128)
        self.d4 = _UpSample(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
        # Apply sigmoid to the output
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return self.tanh(outputs)


if __name__ == "__main__":
    disc = BasicDiscriminator()
    tensor = torch.randn(8, 1,256, 256)
    print(disc(tensor).shape)