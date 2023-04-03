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


def normalize_factor(wi):
    B, N, _, _, _ = wi.shape

    # Calculate L2 norm along dimensions 0, 1, and 2
    l2_norm = torch.sqrt(torch.sum(wi**2, dim=[2, 3, 4]))

    # Compute the element-wise maximum between the L2 norm tensor and the scalar value 1e-4
    result = torch.maximum(l2_norm, torch.tensor(1e-4))

    return result.reshape(B, N, 1, 1, 1)


class ContextualAttention(nn.Module):
    def __init__(self, lambda_value=10.0, rate: float = 2.0):
        super(ContextualAttention, self).__init__()
        self.patch_size = 3
        self.lambda_value = lambda_value

    def extract_mask(self, mask):
        B, _, H, W = mask.shape

        mask_patches = self.to_patches(mask)

        mask_patches = mask_patches.mean(dim=(1, 2, 3), keepdim=True)

        # Compare the mask with zero and convert back to float
        mask_patches = (mask_patches != 0).float()

        mask_patches = mask_patches.view(B, H * W, 1, 1)

        return mask_patches

    def to_patches(self, x):
        B, N, _, _ = x.shape

        # Extract all patches using F.unfold
        patches = F.unfold(x, kernel_size=self.patch_size, padding=1)
        patches = patches.view(B, N, self.patch_size, self.patch_size, -1)

        return patches

    def to_filters(self, x):
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def apply_conv_kernels(self, foreground, background_filters):
        B, _, H, W = foreground.shape
        _, NF, C, H2, W2 = background_filters.shape
        padding = (H2 - 1) // 2, (W2 - 1) // 2

        output_images = []

        for b in range(B):
            # Rearrange background_filters to (NF, C, H2, W2) shape for the current batch
            background_filters_batch = background_filters[b].view(NF, C, H2, W2)

            # Perform convolution
            conv_result = F.conv2d(
                foreground[b].unsqueeze(0), background_filters_batch, padding=padding
            )
            # Shape: (1, NF, H1, W1)

            output_images.append(conv_result)

        # Concatenate the output images along the batch dimension
        output = torch.cat(output_images, dim=0)  # Shape: (B, NF, H1, W1)

        return output

    # def apply_conv_kernels(self, foreground, background_filters):
    #     B, _, H, W = foreground.shape

    #     output_images = []
    #     for b in range(B):
    #         output_patches = []
    #         for l in range(H * W):
    #             kernel = background_filters[b, l].unsqueeze(0)  # Shape: (1, N, 3, 3)
    #             foreground_patch = F.conv2d(
    #                 foreground[b].unsqueeze(0), kernel, padding=1
    #             )  # Shape: (1, 1, H, W)
    #             output_patches.append(foreground_patch)
    #         output_images.append(
    #             torch.cat(output_patches, dim=1)
    #         )  # Shape: (1, L, H, W)
    #     x = torch.cat(output_images, dim=0)  # Shape: (B, L, H, W)
    #     import pdb

    #     pdb.set_trace()
    #     return x

    def apply_deconv_kernels(self, foreground, background_filters):
        B, _, H, W = foreground.shape
        _, NF, C, H2, W2 = background_filters.shape
        padding = (H2 - 1) // 2, (W2 - 1) // 2

        output_images = []

        for b in range(B):
            # Rearrange background_filters to (NF, C, H2, W2) shape for the current batch
            background_filters_batch = background_filters[b].view(NF, C, H2, W2)

            # Perform convolution
            conv_result = F.conv_transpose2d(
                foreground[b].unsqueeze(0), background_filters_batch, padding=padding
            )
            # Shape: (1, NF, H1, W1)

            output_images.append(conv_result)

        # Concatenate the output images along the batch dimension
        output = torch.cat(output_images, dim=0)  # Shape: (B, NF, H1, W1)

        return output

    def forward(self, foreground, background, mask):
        # Since the original paper's code is mostly unreadable
        # we decided to follow the steps in section 4.1 verbatim
        mask_patches = self.extract_mask(mask)
        background_filters = self.to_filters(self.to_patches(background))

        # Find the normalization factor
        normalization_factor = normalize_factor(background_filters)
        background_filters = background_filters / normalization_factor

        # Apply a convolution to the foreground image, which will be used as the query
        attention_map = self.apply_conv_kernels(foreground, background_filters)

        # Now apply softmax to the attention map
        attention_map *= mask_patches
        attention_map = F.softmax(attention_map * self.lambda_value, dim=1)
        attention_map *= mask_patches

        # Apply transpose convolution to the attention map
        attention_map = self.apply_deconv_kernels(attention_map, background_filters)

        return attention_map


class RefineUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int = 128) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            ConvBlock(in_channels, 128, 3, 1, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
        )

    def forward(self, x):
        x = self.refine(x)
        return x


class AttentiveEncoderDecoderNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        context_softmax_scale: float = 10.0,
    ):
        super().__init__()

        self.contracting_path = nn.Sequential(
            ConvBlock(in_channels, 32, 5, 1, 2, activation="elu"),
            ConvBlock(32, 32, 3, 2, 1, activation="elu"),
            ConvBlock(32, 64, 3, 1, 1, activation="elu"),
            ConvBlock(64, 128, 3, 2, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="elu"),
            ConvBlock(128, 128, 3, 1, 1, activation="relu"),
        )

        self.contextual_attention = ContextualAttention(
            lambda_value=context_softmax_scale, rate=2.0
        )

        self.refine_upsampling_path = RefineUpsampleBlock()

    def forward(self, x, mask):
        x = self.contracting_path(x)
        x = self.contextual_attention(x, x, mask)
        x = self.refine_upsampling_path(x)
        return x


class ExpandingPath(nn.Module):
    def __init__(self, out_channels: int):
        super(ExpandingPath, self).__init__()

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


class CoarseNetwork(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.encoder = EncodingConvolutionalPath(in_channels, with_tunning=True)
        self.decoder = ExpandingPath(out_channels)

    def forward(self, x):
        _, x = self.encoder(x)
        x = self.decoder(x)
        return x


class RefinementNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        context_softmax_scale: float = 10.0,
    ) -> None:
        super().__init__()

        # Define the convolutional branch
        self.convolution_branch = EncodingConvolutionalPath(
            in_channels, with_tunning=False
        )

        # Define the attention branch
        self.attention_branch = AttentiveEncoderDecoderNetwork(
            in_channels, context_softmax_scale=context_softmax_scale
        )

        # Define mechanism to refine the upsample performed by the attention branch
        self.refine_attention_branch = RefineUpsampleBlock()

        # Mechanism to decode the final output
        self.decoder = ExpandingPath(out_channels)

        # Mechanism to refine the output of the concatenation
        self.refine_concat = RefineUpsampleBlock(in_channels=128 * 2)

    def forward(self, x, mask):
        c_x, x_hallu = self.convolution_branch(x)

        # Resize the mask according to c_x
        scaled_mask = resize_mask_like(mask, c_x)

        # Apply the attention branch
        x = self.attention_branch(x, scaled_mask)
        x = self.refine_attention_branch(x)

        # Concatenate the hallucinated convolutional branch with the attention branch
        x = torch.cat([x_hallu, x], dim=1)
        x = self.refine_concat(x)

        # Decode the final output
        x = self.decoder(x)

        return x


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
