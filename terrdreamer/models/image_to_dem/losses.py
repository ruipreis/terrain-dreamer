import torch
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.loss_object = nn.BCEWithLogitsLoss()

    def forward(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(
            torch.ones_like(disc_real_output), disc_real_output
        )
        generated_loss = self.loss_object(
            torch.zeros_like(disc_generated_output), disc_generated_output
        )
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss


class GeneratorLoss(nn.Module):
    def __init__(self, loss_lambda: float):
        super().__init__()
        self._lambda = loss_lambda

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, disc_generated_output, gen_output, target):
        gen_loss = self.bce_loss(
            torch.ones_like(disc_generated_output), disc_generated_output
        )
        l1_loss = self.l1_loss(gen_output, target) * self._lambda
        total_gen_loss = gen_loss + l1_loss
        return total_gen_loss
