import torch
import torch.nn as nn
import scipy.ndimage as ndimage


import numpy as np
import matplotlib.pyplot as plt


def visualize_weight_map(height, width, bbox, gamma=0.99, filename="weight_map.png"):
    mask, weight_mask = create_weight_mask(
        height, width, bbox, gamma=gamma, return_mask=True
    )
    weight_mask = (weight_mask - weight_mask.min()) / (
        weight_mask.max() - weight_mask.min()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(mask, cmap="gray")
    ax1.set_title("Binary Mask")
    ax1.axis("off")

    im = ax2.imshow(weight_mask, cmap="viridis")
    ax2.set_title("Weight Map")
    ax2.axis("off")

    fig.colorbar(im, ax=ax2)

    # Save the plot to the specified file
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_weight_mask(
    height, width, bbox, gamma=0.99, return_mask: bool = False, device: str = "cpu"
):
    # Create a mask of zeros with the same size as the input image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Extract the bounding box coordinates
    top, left, bottom, right = bbox

    # Fill the mask with ones inside the bounding box
    mask[top:bottom, left:right] = 1

    # Find the distance transform using the input mask
    distance_transform = ndimage.distance_transform_edt(mask)

    # Compute the weight mask M using the distance_transform and gamma
    weight_mask = gamma**distance_transform
    weight_mask = torch.tensor(weight_mask, dtype=torch.float32, device=device).detach()

    if return_mask:
        return mask, weight_mask

    # Convert the weight mask back to a PyTorch tensor
    return weight_mask


class WeightedL1Loss(nn.Module):
    def __init__(self, normalize_by_mask=True):
        super(WeightedL1Loss, self).__init__()
        self.normalize_by_mask = normalize_by_mask

    def forward(self, input, target, mask):
        # Calculate the L1 loss between the input and target
        l1_loss = torch.mean(torch.abs(input - target) * mask)

        if self.normalize_by_mask:
            # Normalize the loss by the mean of the mask
            l1_loss /= torch.mean(mask)

        return l1_loss


class WGAN_Loss:
    def loss_discriminator(self, critic_real, critic_fake):
        # Calculate the loss for the discriminator
        loss = -(torch.mean(critic_real) - torch.mean(critic_fake))

        return loss

    def loss_generator(self, critic_fake):
        # Calculate the loss for the generator
        loss = -torch.mean(critic_fake)

        return loss


class WGAN_GradientPenalty(nn.Module):
    def __init__(self, lambda_gp=10.0):
        super(WGAN_GradientPenalty, self).__init__()
        self.lambda_gp = lambda_gp

    def forward(self, critic, real, fake, mask):
        # Calculate the gradient penalty loss
        gradient_penalty = self.gradient_penalty(critic, real, fake, mask)

        # Return the gradient penalty loss
        return gradient_penalty

    def gradient_penalty(self, critic, real, fake, mask):
        # Calculate the interpolation
        alpha = torch.rand(real.size(0), 1, 1, 1).to(real.device)
        interpolated = alpha * real + (1 - alpha) * fake

        # Calculate critic scores
        mixed_scores = critic(interpolated)

        # Take the gradient of the scores with respect to the
        # images
        gradient = torch.autograd.grad(
            inputs=interpolated,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Make sure only the valid pixels contribute to the loss
        gradient = gradient * mask

        # Flatten the gradient
        gradient = gradient.view(gradient.size(0), -1)

        # Calculate the magnitude of the gradient
        gradient_norm = gradient.norm(2, dim=1)

        # Penalize the mean squared distance of the gradient from 1
        gradient_penalty = self.lambda_gp * torch.mean((gradient_norm - 1) ** 2)

        return gradient_penalty


if __name__ == "__main__":
    visualize_weight_map(500, 500, (50, 50, 300, 300), gamma=0.99)
