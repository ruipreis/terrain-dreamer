import torch
import torch.nn as nn
from terrdreamer.models.image_to_dem import DEM_Pix2Pix


# Masked pix2pix uses a masked l1 loss
class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, input, target, mask):
        # Calculate the L1 loss in the masked region
        loss = torch.abs(input - target) * (mask / 2 + 0.5)
        loss = torch.mean(loss)

        return loss


class Masked_Pix2Pix(DEM_Pix2Pix):
    def __init__(
        self,
        lambda_l1=100,
        ngf: int = 64,
        ndf: int = 64,
        lambda_gp: float = 10,
    ):
        super().__init__(
            4,
            3,
            lambda_l1=lambda_l1,
            ngf=ngf,
            ndf=ndf,
            loss="wgangp",
            lambda_gp=lambda_gp,
        )

        self.l1_loss = MaskedL1Loss()
