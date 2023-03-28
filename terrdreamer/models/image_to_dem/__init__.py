import torch
import torch.nn as nn

from terrdreamer.models.image_to_dem.base_models import (
    BasicDiscriminator,
    UNetGenerator,
)

# Some tricks were adopted from the followings tips:
# https://github.com/soumith/ganhacks/blob/master/README.md


class DEM_Pix2Pix:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lambda_l1=100,
        ngf: int = 64,
        ndf: int = 64,
        # Label smoothing - this allows to not be too confident about the labels
        # According to https://github.com/soumith/ganhacks/issues/10, label smoothing
        # should only occur in the discriminator
        label_smoothing: bool = True,
        label_smoothing_factor: float = 0.1,
        # Used for wasserstein gradient penalty
        loss: str = "vanilla",  # Add loss argument
        lambda_gp: float = 10,  # Add lambda_gp argument for gradient penalty weight
        # Indicates if the model should be run on inference, if this is the case
        # only the generator will be loaded and no loss.
        inference: bool = False,
    ):
        self.generator = UNetGenerator(in_channels, out_channels, d=ngf)

        if not inference:
            self.discriminator = BasicDiscriminator(in_channels + out_channels, d=ndf)
            self.label_smoothing = label_smoothing
            self.label_smoothing_factor = label_smoothing_factor
            self.loss = loss
            self.l1_loss = nn.L1Loss()
            self.l1_lambda = lambda_l1

            if loss == "wgangp":
                self.lambda_gp = lambda_gp
            else:
                self.bce_loss = nn.BCEWithLogitsLoss()

    def prepare_discriminator_step(self):
        self.discriminator.set_requires_grad(True)
        self.generator.set_requires_grad(False)
        self.discriminator.zero_grad()

    def prepare_generator_step(self):
        self.discriminator.set_requires_grad(False)
        self.generator.set_requires_grad(True)
        self.generator.zero_grad()

    # Add gradient_penalty function
    def gradient_penalty(self, x, real_images, fake_images, device):
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolates = (
            (alpha * real_images + (1 - alpha) * fake_images)
            .detach()
            .requires_grad_(True)
        )
        d_interpolates = self.discriminator(x, interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def step_discriminator(self, x, y, optimizer):
        fakeY = self.generator(x)
        real_AB = self.discriminator(x, y)
        fake_AB = self.discriminator(x, fakeY.detach())

        if self.loss == "wgangp":
            real_loss = -torch.mean(real_AB)
            fake_loss = torch.mean(fake_AB)
            loss = (
                fake_loss
                + real_loss
                + self.lambda_gp * self.gradient_penalty(x, y, fakeY.detach(), x.device)
            )
        else:
            # Get the expected real labels
            real_labels = torch.ones_like(real_AB).detach()

            # If label smoothing is enabled, we'll make the labels a bit smaller
            if self.label_smoothing:
                real_labels = real_labels.fill_(1 - self.label_smoothing_factor)

            real_loss = self.bce_loss(real_AB, real_labels)
            fake_loss = self.bce_loss(fake_AB, torch.zeros_like(fake_AB))
            loss = (real_loss + fake_loss) / 2

        loss.backward()
        optimizer.step()

        return loss.item(), real_loss.item(), fake_loss.item()

    def step_generator(self, x, y, optimizer, *args):
        fakeY = self.generator(x)
        fake_AB = self.discriminator(x, fakeY)

        if self.loss == "wgangp":
            loss = -torch.mean(fake_AB)
            bce_loss = None
        else:
            bce_loss = self.bce_loss(fake_AB, torch.ones_like(fake_AB))
            loss = bce_loss

        l1_loss = self.l1_loss(fakeY, y, *args) * self.l1_lambda
        loss += l1_loss

        loss.backward()
        optimizer.step()

        return (
            loss.item(),
            (bce_loss.item() if bce_loss is not None else None),
            l1_loss.item(),
        )

    def test(self, x, y, *args):
        fakeY = self.generator(x)
        fake_AB = self.discriminator(x, fakeY)
        real_AB = self.discriminator(x, y)

        if self.loss == "wgangp":
            D_loss = torch.mean(fake_AB) - torch.mean(real_AB)
            G_loss = -torch.mean(fake_AB)
        else:
            D_loss = self.bce_loss(fake_AB, torch.zeros_like(fake_AB)) + self.bce_loss(
                real_AB, torch.ones_like(real_AB)
            )
            G_loss = self.bce_loss(fake_AB, torch.ones_like(fake_AB))

        G_loss = G_loss + self.l1_loss(fakeY, y, *args) * self.l1_lambda

        return fakeY, D_loss, G_loss

    def load_generator(self, path):
        self.generator.load_state_dict(torch.load(path))

    def load_discriminator(self, path):
        self.discriminator.load_state_dict(torch.load(path))

    def weight_init(self, mean, std):
        self.generator.weight_init(mean, std)
        self.discriminator.weight_init(mean, std)

    def to(self, device):
        self.generator.to(device)
        self.discriminator.to(device)

    def save(self, epoch_no: int):
        torch.save(self.generator.state_dict(), f"generator_{epoch_no}.pt")
        torch.save(self.discriminator.state_dict(), f"discriminator_{epoch_no}.pt")
