import torch
import torch.nn as nn

from terrdreamer.models.image_to_dem.base_models import BasicDiscriminator, UNetGenerator


class DEM_Pix2Pix:
    def __init__(self, in_channels:int, out_channels:int, lambda_l1=100, ngf:int=64, ndf:int=64):
        self.generator = UNetGenerator(in_channels, out_channels, d=ngf)
        self.discriminator = BasicDiscriminator(in_channels+out_channels, d=ndf)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.l1_lambda = lambda_l1

    def prepare_discriminator_step(self):
        self.discriminator.set_requires_grad(True)
        self.generator.set_requires_grad(False)
        self.discriminator.zero_grad()

    def prepare_generator_step(self):
        self.discriminator.set_requires_grad(False)
        self.generator.set_requires_grad(True)
        self.generator.zero_grad()

    def step_discriminator(self, x, y, optimizer):
        fakeY = self.generator(x)
        real_AB = self.discriminator(x, y)
        fake_AB = self.discriminator(x, fakeY.detach())

        real_loss = self.bce_loss(real_AB, torch.ones_like(real_AB))
        fake_loss = self.bce_loss(fake_AB, torch.zeros_like(fake_AB))
        loss = (real_loss + fake_loss) / 2

        loss.backward()
        optimizer.step()

        return loss, real_loss.item(), fake_loss.item()
    
    def step_generator(self, x, y, optimizer):
        fakeY = self.generator(x)
        fake_AB = self.discriminator(x, fakeY)

        bce_loss = self.bce_loss(fake_AB, torch.ones_like(fake_AB)) 
        l1_loss = self.l1_loss(fakeY, y)*self.l1_lambda
        loss = bce_loss + l1_loss

        loss.backward()
        optimizer.step()

        return loss, bce_loss.item(), l1_loss.item()
    
    def test(self, x, y):
        fakeY = self.generator(x)
        fake_AB = self.discriminator(x, fakeY)
        real_AB = self.discriminator(x, y)

        D_loss = self.bce_loss(fake_AB, torch.zeros_like(fake_AB)) + self.bce_loss(real_AB, torch.ones_like(real_AB))
        G_loss = self.bce_loss(fake_AB, torch.ones_like(fake_AB)) + self.l1_loss(fakeY, y)*self.l1_lambda

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

    def save(self, epoch_no:int):
        torch.save(self.generator.state_dict(), f"generator_{epoch_no}.pt")
        torch.save(self.discriminator.state_dict(), f"discriminator_{epoch_no}.pt")