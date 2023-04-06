import torch
import torch.nn as nn
from pathlib import Path
from terrdreamer.models.infinity_grid.layers import CoarseNetwork, RefinementNetwork
from terrdreamer.models.infinity_grid.critics import LocalCritic, GlobalCritic
from terrdreamer.models.infinity_grid.loss import (
    WGAN_GradientPenalty,
    WeightedL1Loss,
    WGAN_Loss,
    create_weight_mask,
)


def local_patch(x, bbox):
    """
    Crop local patch according to bbox.
    Args:
        x: input
        bbox: (top, left, height, width)
    Returns:
        torch.Tensor: local patch
    """
    top, left, height, width = bbox
    patch = x[:, :, top : top + height, left : left + width]
    return patch


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
        scale_factor: float,
        # Gamma for the discounting mask creation
        gamma: float = 0.99,
        # Indicates weight of gradient penalty
        lambda_gp: float = 10.0,
        # Indicates if the model should be run on inference, if this is the case
        # only the generator will be loaded and no loss.
        inference: bool = False,
    ):
        self.inpaint_generator = InpaintCAModel()
        self.gamma = gamma

        if not inference:
            self.global_critic = GlobalCritic(height, width)
            self.local_critic = LocalCritic(
                int(height * scale_factor), int(width * scale_factor)
            )

            # Get the loss functions for the local and global critic
            self.wgan_loss = WGAN_Loss()
            self.wgan_gradient_penalty = WGAN_GradientPenalty(lambda_gp=lambda_gp)

            # Spatial discounted loss uses L1 loss both without by mask normalization
            self.l1_loss = WeightedL1Loss(normalize_by_mask=False)
            self.ae_loss = WeightedL1Loss(normalize_by_mask=True)

    def sample_discriminator(self, x, x_complete, local_x, local_x_complete):
        # Concatenate the variables that refers to local and global regions
        # and pass them through the discriminator
        x_discriminator = torch.cat([x, x_complete], dim=0)
        local_x_discriminator = torch.cat([local_x, local_x_complete], dim=0)

        # Pass the concatenated tensors through the discriminators
        global_critic_out = self.global_critic(x_discriminator)
        local_critic_out = self.local_critic(local_x_discriminator)

        # Unroll to get the output for the real and fake images
        real_global, fake_global = torch.split(
            global_critic_out, global_critic_out.size(0) // 2, dim=0
        )
        real_local, fake_local = torch.split(
            local_critic_out, local_critic_out.size(0) // 2, dim=0
        )

        return real_global, fake_global, real_local, fake_local

    def step_discriminator(self, x, mask, bbox, local_optimizer, global_optimizer):
        # Mask out the input image
        x_incomplete = x * (1 - mask)

        # Run through the inpaint generator
        _, x2 = self.inpaint_generator(x_incomplete, mask)
        x_predicted = x2

        # Now create a new tensor that has the original image where the mask is not
        # and the inpainted image where the mask is
        x_complete = x_predicted * mask + x_incomplete * (1 - mask)

        # Extract subregions from the various tensors
        local_x = local_patch(x, bbox)
        local_x_complete = local_patch(x_complete, bbox)
        local_mask = local_patch(mask, bbox)

        # Pass through both discriminators
        real_global, fake_global, real_local, fake_local = self.sample_discriminator(
            x, x_complete, local_x, local_x_complete
        )

        # Compute the loss for the joint discriminator
        d_loss_global = self.wgan_loss.loss_discriminator(real_global, fake_global)
        d_loss_local = self.wgan_loss.loss_discriminator(real_local, fake_local)
        d_loss = d_loss_global + d_loss_local

        # Compute the gradient penalty for the joint discriminator
        gp_global = self.wgan_gradient_penalty(
            self.global_critic, x, x_complete.detach(), mask
        )
        gp_local = self.wgan_gradient_penalty(
            self.local_critic, local_x, local_x_complete, local_mask
        )
        gp = gp_global + gp_local

        # Compute the total loss
        loss = d_loss + gp

        # Backpropagate the loss
        loss.backward()
        local_optimizer.step()
        global_optimizer.step()

        return {
            "d_loss": d_loss.item(),
            "gp": gp.item(),
            "loss": loss.item(),
        }

    def eval(self, masked_x, mask):
        return self.inpaint_generator(masked_x, mask)

    def step_generator(self, x, mask, bbox, optimizer):
        # Mask out the input image
        x_incomplete = x * (1 - mask)

        # Run through the inpaint generator
        x1, x2 = self.inpaint_generator(x_incomplete, mask)
        x_predicted = x2

        # Now create a new tensor that has the original image where the mask is not
        # and the inpainted image where the mask is
        x_complete = x_predicted * mask + x_incomplete * (1 - mask)

        # Extract subregions from the various tensors
        local_x = local_patch(x, bbox)
        local_x1 = local_patch(x1, bbox)
        local_x_predicted = local_patch(x_predicted, bbox)
        local_x_complete = local_patch(x_complete, bbox)

        # Pass through both discriminators
        fake_global = self.global_critic(x_complete)
        fake_local = self.local_critic(local_x_complete)

        # Create a spatial discounting mask and extract only the local region
        _, _, H, W = x.shape
        local_discounted_mask = local_patch(
            create_weight_mask(H, W, bbox, gamma=self.gamma, device=x.device), bbox
        )

        # Compute the losses

        # WGAN loss - for the local and global critic
        wgan_loss = self.wgan_loss.loss_generator(
            fake_global
        ) + self.wgan_loss.loss_generator(fake_local)

        # Measure the ability of the model to reconstruct the original region, with
        # discounting based on distance to nearest non-null pixel
        l1_loss = self.l1_loss(local_x, local_x1, local_discounted_mask) + self.l1_loss(
            local_x, local_x_predicted, local_discounted_mask
        )

        # Ability of the model to reconstruct the un-painted region
        ae_loss = self.ae_loss(x, x1, 1 - mask) + self.ae_loss(x, x_predicted, 1 - mask)

        # Aggregate the losses
        loss = wgan_loss + l1_loss + ae_loss

        # Apply backprop and update the weights
        loss.backward()
        optimizer.step()

        return {
            "wgan_loss": wgan_loss.item(),
            "l1_loss": l1_loss.item(),
            "ae_loss": ae_loss.item(),
            "loss": loss.item(),
        }

    def prepare_discriminator_step(self):
        # Prepare for the discriminator's step
        self.inpaint_generator.set_requires_grad(False)
        self.local_critic.set_requires_grad(True)
        self.global_critic.set_requires_grad(True)
        self.local_critic.zero_grad()
        self.global_critic.zero_grad()

    def prepare_generator_step(self):
        # Prepare for the generator's step
        self.inpaint_generator.set_requires_grad(True)
        self.local_critic.set_requires_grad(False)
        self.global_critic.set_requires_grad(False)
        self.inpaint_generator.zero_grad()

    def save(self, save_dir: Path):
        torch.save(
            self.inpaint_generator.state_dict(),
            save_dir / "inpaint_generator.pth",
        )
        torch.save(
            self.local_critic.state_dict(),
            save_dir / "inpaint_local_critic.pth",
        )
        torch.save(
            self.global_critic.state_dict(),
            save_dir / "inpaint_global_critic.pth",
        )

    def load_pretrained_if_needed(
        self,
        pretrained_generator_path,
        pretrained_local_critic_path,
        pretrained_global_critic_path,
    ):
        if pretrained_generator_path is not None:
            self.inpaint_generator.load_state_dict(
                torch.load(pretrained_generator_path)
            )

        if pretrained_local_critic_path is not None:
            self.local_critic.load_state_dict(torch.load(pretrained_local_critic_path))

        if pretrained_global_critic_path is not None:
            self.global_critic.load_state_dict(
                torch.load(pretrained_global_critic_path)
            )

    def to(self, device):
        self.inpaint_generator.to(device)
        self.local_critic.to(device)
        self.global_critic.to(device)


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
