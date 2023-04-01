import torch
import torch.nn as nn

from terrdreamer.models.infinity_grid.layers import EncoderDecoderNetwork

class DeepFillV1(nn.Module):
    # This is based on the following paper:
    # https://arxiv.org/abs/1801.07892
    #
    # Which includes the following general structure:
    # 1. Coarse network
    # 2. Refinement network
    # 3. Local Discriminator
    # 4. Global Discriminator
    # 5. Use of contextual attention

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        self.coarse_network = EncoderDecoderNetwork(
            in_channels=in_channels+1+1, # 1 for mask, 1 for depth
            out_channels=out_channels,
        )
        
    def forward(self, x, mask):
        # Keep a variable referencing the original image
        xin = x
        
        # Concatenate the mask and the image
        B, _, H, W = x.shape
        x_ones = torch.ones(B, 1, H, W)
        x = torch.cat([x, x_ones, x_ones * mask], dim=1)

        x = self.coarse_network(x)
        
        return
        