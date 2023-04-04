import torch
import torch.nn as nn
from pathlib import Path
from terrdreamer.models.infinity_grid.layers import CoarseNetwork, RefinementNetwork
from terrdreamer.models.infinity_grid.critics import (
    LocalCritic,
    GlobalCritic,
    GeneralCritic,
)
from terrdreamer.models.infinity_grid.loss import (
    WGAN_GradientPenalty,
    WeightedL1Loss,
    WGAN_Loss,
)


class InpaintCAModel(nn.Module):
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
        super(InpaintCAModel, self).__init__()

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
        x_ones = torch.ones(B, 1, H, W).to(x.device)
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

    def set_requires_grad(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad = requires_grad


class DeepFillV1:
    def __init__(
        self,
        height: int,
        width: int,
        # Indicates weight of gradient penalty
        lambda_gp: float = 10.0,
        # Indicates if the model should be run on inference, if this is the case
        # only the generator will be loaded and no loss.
        inference: bool = False,
    ):
        self.inpaint_generator = InpaintCAModel()

        if not inference:
            self.critic = GeneralCritic(height, width)

            # Get the loss functions for the local and global critic
            self.wgan_loss = WGAN_Loss()
            self.wgan_gradient_penalty = WGAN_GradientPenalty(lambda_gp=lambda_gp)

            # Spatial discounted loss uses L1 loss both without by mask normalization
            self.l1_loss = WeightedL1Loss(normalize_by_mask=False)
            self.ae_loss = WeightedL1Loss(normalize_by_mask=True)

    def step_discriminator(self):
        raise NotImplementedError

    def step_generator(self):
        raise NotImplementedError

    def prepare_discriminator_step(self):
        # Prepare for the discriminator's step
        self.inpaint_generator.set_requires_grad(False)
        self.critic.set_requires_grad(True)
        self.critic.zero_grad()

    def prepare_generator_step(self):
        # Prepare for the generator's step
        self.inpaint_generator.set_requires_grad(True)
        self.critic.set_requires_grad(False)
        self.critic.zero_grad()

    def save(self, save_dir: Path):
        torch.save(
            self.inpaint_generator.state_dict(),
            save_dir / "inpaint_generator.pth",
        )
        torch.save(
            self.critic.state_dict(),
            save_dir / "inpaint_critic.pth",
        )

    def load_pretrained_if_needed(
        self,
        pretrained_generator_path,
        pretrained_discriminator_path,
    ):
        if pretrained_generator_path is not None:
            self.inpaint_generator.load_state_dict(
                torch.load(pretrained_generator_path)
            )

        if pretrained_discriminator_path is not None:
            self.critic.load_state_dict(torch.load(pretrained_discriminator_path))

    def to(self, device):
        self.inpaint_generator.to(device)
        self.critic.to(device)


if __name__ == "__main__":
    deepfill = InpaintCAModel().to("cuda")
    rand_input = torch.rand(16, 3, 256, 256).to("cuda")
    rand_mask = torch.zeros(16, 1, 256, 256).to("cuda")
    rand_mask[:, :, 100:150, 100:150] = 1
    import time

    start = time.time()
    x_stage1, x_stage2 = deepfill(rand_input, rand_mask)
    end = time.time()
    print(f"Time taken: {end - start}")
    print(x_stage1.shape)
    print(x_stage2.shape)
