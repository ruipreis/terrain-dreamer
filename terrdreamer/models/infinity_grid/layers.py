import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_mask_like(mask, x):
    """
    Resize mask to have the same shape as x.
    Args:
        mask (torch.Tensor): Original mask.
        x (torch.Tensor): Reference tensor for the target shape.
    Returns:
        torch.Tensor: Resized mask
    """
    # Get the target height and width from the reference tensor x
    target_height, target_width = x.shape[-2:]

    # Resize the mask using the interpolate function with mode 'nearest' for nearest-neighbor interpolation
    mask_resize = F.interpolate(
        mask, size=(target_height, target_width), mode="nearest"
    )

    return mask_resize


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


class EncodingConvolutionalPath(nn.Module):
    def __init__(self, in_channels: int, with_tunning: bool = True) -> None:
        super(EncodingConvolutionalPath, self).__init__()
        self.with_tunning = with_tunning

        self.contracting_path = nn.Sequential(
            ConvBlock(in_channels, 32, 5, 1, 2, activation="elu"),
            ConvBlock(32, 64, 3, 2, 1, activation="elu"),
            ConvBlock(64, 64, 3, 1, 1, activation="elu"),
            ConvBlock(64, 128, 3, 2, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
        )

        self.dilation_path = nn.Sequential(
            ConvBlock(128, 128, 3, 1, 2, dilation=2, activation="elu"),
            ConvBlock(128, 128, 3, 1, 4, dilation=4, activation="elu"),
            ConvBlock(128, 128, 3, 1, 8, dilation=8, activation="elu"),
            ConvBlock(128, 128, 3, 1, 16, dilation=16, activation="elu"),
        )

        if self.with_tunning:
            self.tunning_path = nn.Sequential(
                ConvBlock(128, 128, 3, 1, 1, activation="elu"),
                ConvBlock(128, 128, 3, 1, 1, activation="elu"),
            )

    def forward(self, x):
        c_x = self.contracting_path(x)
        x = self.dilation_path(c_x)

        if self.with_tunning:
            x = self.tunning_path(x)

        return c_x, x


class ContextualAttention(nn.Module):
    def __init__(self, patch_size=3, lambda_value=10.0, rate: float = 2.0):
        super(ContextualAttention, self).__init__()
        self.patch_size = patch_size
        self.lambda_value = lambda_value
        self.rate = rate

    def forward(self, foreground, background, mask=None):
        # Resize the rate of foreground, background and mask
        foreground = F.interpolate(
            foreground, scale_factor=1 / self.rate, mode="nearest"
        )
        background = F.interpolate(
            background, scale_factor=1 / self.rate, mode="nearest"
        )

        if mask is not None:
            mask = F.interpolate(mask, scale_factor=1 / self.rate, mode="nearest")
            foreground = foreground * mask

        # Extract patches from background
        b_patches = F.unfold(
            background, kernel_size=self.patch_size, padding=self.patch_size // 2
        )

        # Reshape patches as convolutional filters
        b_patches = b_patches.view(
            background.size(0), background.size(1), self.patch_size, self.patch_size, -1
        )

        # Match foreground patches with background ones
        similarity_list = []
        for i in range(b_patches.size(-1)):
            patch_i = b_patches[..., i]
            similarity_i = F.conv2d(foreground, patch_i, padding=self.patch_size // 2)
            similarity_list.append(similarity_i)

        similarity = torch.cat(similarity_list, dim=1)

        # Normalize with cosine similarity
        similarity = F.normalize(
            similarity.view(similarity.size(0), -1), dim=1
        ).view_as(similarity)

        # Weigh similarity with scaled softmax along x0, y0 dimensions
        attention_score = F.softmax(self.lambda_value * similarity, dim=1)

        # Reuse extracted background patches as deconvolutional filters to reconstruct foregrounds
        foreground_size = foreground.size()
        reconstructed_foreground = torch.zeros_like(foreground)
        for i in range(attention_score.size(1)):
            score_i = attention_score[:, i].view(
                foreground_size[0], 1, foreground_size[2], foreground_size[3]
            )
            patch_i = b_patches[..., i]
            reconstructed_i = F.conv_transpose2d(
                score_i, patch_i, padding=self.patch_size // 2
            )
            reconstructed_foreground += reconstructed_i

        if mask is not None:
            reconstructed_foreground = reconstructed_foreground * mask

        # Resize the rate of reconstructed foreground
        reconstructed_foreground = F.interpolate(
            reconstructed_foreground, scale_factor=self.rate, mode="nearest"
        )

        return reconstructed_foreground


class AttentiveEncoderDecoderNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        context_softmax_scale: float = 10.0,
    ):
        self.contracting_path = nn.Sequential(
            ConvBlock(in_channels, 32, 5, 1, 2, activation="elu"),
            ConvBlock(32, 32, 3, 2, 1, activation="elu"),
            ConvBlock(32, 64, 3, 1, 1, activation="elu"),
            ConvBlock(64, 128, 3, 2, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="relu"),
        )

        self.contextual_attention = ContextualAttention(
            patch_size=3, lambda_value=context_softmax_scale, rate=2.0
        )

        self.refine_upsampling_path = nn.Sequential(
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
        )

    def forward(self, x, mask):
        x = self.contracting_path(x)
        x = self.contextual_attention(x, x, mask)
        x = self.refine_upsampling_path(x)
        return x


class ContractingPath(nn.Module):
    def __init__(self, out_channels: int):
        super(ContractingPath, self).__init__()

        # Assumed to be the output with tunning from the encoder
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
        x = self.decoder(x)
        return torch.clip(x, -1.0, 1.0)


class SimpleEncoderDecoderNetwork(nn.Module):
    pass


if __name__ == "__main__":
    import torch

    # Example usage:
    foreground = torch.randn(
        1, 3, 64, 64
    )  # Example foreground tensor (Batch x Channels x Height x Width)
    background = torch.randn(
        1, 3, 256, 256
    )  # Example background tensor (Batch x Channels x Height x Width)

    contextual_attention = ContextualAttention()
    reconstructed_foreground = contextual_attention(foreground, background)

    import pdb

    pdb.set_trace()
