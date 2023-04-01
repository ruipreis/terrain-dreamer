import torch
import torch.nn as nn

from terrdreamer.models.infinity_grid.layers import CoarseNetwork, RefinementNetwork


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
        context_softmax_scale: float = 10.0,
    ):
        super(DeepFillV1, self).__init__()

        # Define the coarse network
        self.coarse_network = CoarseNetwork(
            in_channels=in_channels + 1 + 1,  # 1 for mask, 1 for depth
            out_channels=out_channels,
        )

        # Define the refinement network
        self.refinement_network = RefinementNetwork(
            in_channels=in_channels + 1 + 1,  # 1 for mask, 1 for depth
            out_channels=out_channels,
            context_softmax_scale=context_softmax_scale,
        )

    def prepare_input(self, x, mask):
        # Concatenate the mask and the image
        B, _, H, W = x.shape
        x_ones = torch.ones(B, 1, H, W)
        x = torch.cat([x, x_ones, x_ones * mask], dim=1)
        return x

    def forward(self, x, mask):
        # Keep a variable referencing the original image
        x_in = x

        # Pass the image through the coarse network
        x = self.prepare_input(x, mask)
        x_stage1 = self.coarse_network(x)

        # Pass the image through the refinement network
        x = x_stage1 * mask + x_in * (1 - mask)
        x = self.prepare_input(x, mask)
        x_stage2 = self.refinement_network(x, mask)

        return x_stage1, x_stage2


if __name__ == "__main__":
    deepfill = DeepFillV1()
    rand_input = torch.rand(1, 3, 256, 256)
    rand_mask = torch.zeros(1, 1, 256, 256)
    rand_mask[:, :, 100:150, 100:150] = 1

    import time

    start = time.time()
    x_stage1, x_stage2 = deepfill(rand_input, rand_mask)
    end = time.time()
    print(f"Time taken: {end - start}")
