import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class CustomUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomUpsample, self).__init__()
        self._upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self._conv = nn.Conv2d(in_channels, out_channels, 4,2,1)
    
    def forward(self, x):
        x = self._upsample(x)
        x = self._conv(x)
        return x
        

class UNetGenerator(nn.Module):
    # Just like the original paper
    def __init__(self, in_channels:int, out_channels:int, d: int = 64):
        # The number of channels is expected to be the sum of the number of channels of the input and the target
        super().__init__()
                # Unet encoder
        self.conv1 = nn.Conv2d(in_channels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)

        # Unet decoder
        self.deconv1 = CustomUpsample(d * 8, d * 8)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = CustomUpsample(d * 8 * 2, d * 8)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = CustomUpsample(d * 8 * 2, d * 8)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = CustomUpsample(d * 8 * 2, d * 8)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = CustomUpsample(d * 8 * 2, d * 4)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = CustomUpsample(d * 4 * 2, d * 2)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = CustomUpsample(d * 2 * 2, d)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = CustomUpsample(d * 2, out_channels)

        # self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8,4,2,1)
        # self.deconv1_bn = nn.BatchNorm2d(d * 8)
        # self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8,4,2,1)
        # self.deconv2_bn = nn.BatchNorm2d(d * 8)
        # self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8,4,2,1)
        # self.deconv3_bn = nn.BatchNorm2d(d * 8)
        # self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8,4,2,1)
        # self.deconv4_bn = nn.BatchNorm2d(d * 8)
        # self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4,4,2,1)
        # self.deconv5_bn = nn.BatchNorm2d(d * 4)
        # self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2,4,2,1)
        # self.deconv6_bn = nn.BatchNorm2d(d * 2)
        # self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d,4,2,1)
        # self.deconv7_bn = nn.BatchNorm2d(d)
        # self.deconv8 = nn.ConvTranspose2d(d * 2, out_channels,4,2,1)
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
        
    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        o = torch.tanh(d8)

        return o


class BasicDiscriminator(nn.Module):
    # initializers
    def __init__(self, input_channels,d=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            self._modules[m].bias.data.zero_()

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x


if __name__ == "__main__":
    input_rgb = torch.randn(8, 3, 256, 256)
    target_dem = torch.randn(8, 1, 256, 256)
    
    disc = BasicDiscriminator(3+1)
    print("Experimenting with the discriminator...")
    print(disc(input_rgb, target_dem).shape)
    
    generator = UNetGenerator(3)
    print("Experimenting with the generator...")
    print(generator(input_rgb).shape)