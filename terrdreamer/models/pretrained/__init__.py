import os

import gdown
import pkg_resources
import torch
import torch.nn as nn

from terrdreamer.dataset import DEM_MAX_ELEVATION, DEM_MIN_ELEVATION
from terrdreamer.models import replace_batchnorm2d_with_instancenorm
from terrdreamer.models.image_to_dem import DEM_Pix2Pix
from terrdreamer.models.infinity_grid import DeepFillV1
from terrdreamer.models.infinity_grid.train import random_bbox
from terrdreamer.models.progan_satelite.progan import Generator


def download_from_google_drive(file_id, output_path):
    gdown.download(id=file_id, output=output_path, quiet=False)


def check_and_download_pretrained_model(model_base_name, file_id):
    resource_path = f"models/pretrained/{model_base_name}"

    # Attempt to find the resource
    res_path = pkg_resources.resource_filename("terrdreamer", resource_path)

    if os.path.isfile(res_path):
        print(f"Using locally cached '{model_base_name}' model")
    else:
        # If the resource is not found, download it
        print(
            f"Model '{model_base_name}' not found in the '{res_path}' subpackage. Downloading..."
        )
        download_from_google_drive(file_id, res_path)
        print(f"Model '{model_base_name}' downloaded to '{res_path}'.")

    return res_path


def convert_dem_batch(dem_batch, repeat=True):
    # Convert the DEM to a 3-channel image
    dem_batch = (dem_batch + 1) / 2

    # Unnormalize the DEM
    dem_batch = dem_batch * (DEM_MAX_ELEVATION - DEM_MIN_ELEVATION) + DEM_MIN_ELEVATION

    # Find the minimum value for each sample in the batch - for all pixels
    min_values = (
        dem_batch.min(dim=1, keepdim=True)[0]
        .min(dim=2, keepdim=True)[0]
        .min(dim=3, keepdim=True)[0]
    )
    max_values = (
        dem_batch.max(dim=1, keepdim=True)[0]
        .max(dim=2, keepdim=True)[0]
        .max(dim=3, keepdim=True)[0]
    )

    # Normalize the DEM
    dem_batch = (dem_batch - min_values) / (max_values - min_values)

    # Put in range (-1, 1)
    dem_batch = dem_batch * 2 - 1

    if repeat:
        # Convert the DEM to a 3-channel image
        dem_batch = dem_batch.repeat(1, 3, 1, 1)

    return dem_batch


class PretrainedImageToDEM(nn.Module):
    GENERATOR_FILE_NAME = "image_to_dem_generator.pt"
    FILE_ID = "1FfRDMAHhiKQh29TDQcGfBPb0x9mZNVkz"

    def __init__(self, use_instance_norm=False):
        super().__init__()
        self.model_path = check_and_download_pretrained_model(
            self.GENERATOR_FILE_NAME, self.FILE_ID
        )

        # Create the model
        self.pix2pix = DEM_Pix2Pix(3, 1, inference=True)

        # Load the pretrained model
        self.pix2pix.load_generator(self.model_path)

        # Make sure the gradient is not computed
        self.pix2pix.generator.set_requires_grad(False)

        if use_instance_norm:
            replace_batchnorm2d_with_instancenorm(self.pix2pix.generator)

    def forward(self, x) -> torch.tensor:
        return self.pix2pix.generator(x)

    def to(self, device):
        self.pix2pix.generator.to(device)
        return self


class PretrainedDEMToImage(nn.Module):
    GENERATOR_FILE_NAME = "dem_to_image_generator.pt"
    FILE_ID = "1aQ0izHbKW7fJ2-FXjg23k6fCcr9uWUM8"

    def __init__(self, use_instance_norm=False):
        super().__init__()
        self.model_path = check_and_download_pretrained_model(
            self.GENERATOR_FILE_NAME, self.FILE_ID
        )

        # Create the model
        self.pix2pix = DEM_Pix2Pix(1, 3, inference=True)

        # Load the pretrained model
        self.pix2pix.load_generator(self.model_path)

        # Make sure the gradient is not computed
        self.pix2pix.generator.set_requires_grad(False)

        if use_instance_norm:
            replace_batchnorm2d_with_instancenorm(self.pix2pix.generator)

    def forward(self, x) -> torch.tensor:
        return self.pix2pix.generator(x)


class PretrainedProGAN(nn.Module):
    GENERATOR_FILE_NAME = "progan_generator.pth"
    FILE_ID = "1okMqM3D35wuFk4ZznYycYEVVF95uyJV2"

    def __init__(self):
        super().__init__()
        self.model_path = check_and_download_pretrained_model(
            self.GENERATOR_FILE_NAME, self.FILE_ID
        )

        # Create the model
        self.progan = Generator(256, 256, 3)

        # Load the pretrained model
        self.progan.load_state_dict(torch.load(self.model_path))

        # Make sure the gradient is not computed
        self.progan.eval()

    def forward(self, x) -> torch.tensor:
        # Alpha = 1, n_steps = 7
        return self.progan(x, 1, 7 - 1)


class PretrainedDeepfillV1:
    GENERATOR_FILE_NAME = "deepfillv1_generator.pth"
    FILE_ID = "16lStlTfhLSNsGFAmRvAWoXaLssGHFobu"

    def __init__(self):
        self.model_path = check_and_download_pretrained_model(
            self.GENERATOR_FILE_NAME, self.FILE_ID
        )

        # Create the model
        self.deepfillv1 = DeepFillV1(inference=True)

        # Load the pretrained model
        self.deepfillv1.inpaint_generator.load_state_dict(torch.load(self.model_path))

        # Make sure the gradient is not computed
        self.deepfillv1.inpaint_generator.set_requires_grad(False)

    def __call__(self, masked_x, mask):
        return self.deepfillv1.eval(masked_x, mask)

    def to(self, device):
        self.deepfillv1.inpaint_generator.to(device)
        return self


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import torch
    import torchvision
    from torch.utils.data import DataLoader

    from terrdreamer.dataset import AW3D30Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-to-dem", action="store_true")
    parser.add_argument("--dem-to-image", action="store_true")
    parser.add_argument("--progan", action="store_true")
    parser.add_argument("--inpainting", action="store_true")
    parser.add_argument("--min-scale-factor", type=float, default=0.3)
    parser.add_argument("--max-scale-factor", type=float, default=0.7)

    # Some of the models were trained with batch normalization, since pix2pix, for example
    # expects batch normalization, we need to replace it with instance normalization. So
    # that we can use a greater batch size.
    parser.add_argument("--use-instancenorm", action="store_true")

    ARGS = parser.parse_args()

    # Grab the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset and the pretrained model
    dataset_loader = DataLoader(
        AW3D30Dataset(ARGS.dataset, normalize=True),
        batch_size=ARGS.batch_size,
        shuffle=True,
    )

    # Get a batch of data
    sat, gtif = next(iter(dataset_loader))

    if ARGS.image_to_dem:
        # When using IMAGE TO DEM and DEM TO IMAGE, it's very important to use
        # batch size of 1, due to the expected instance normalization.
        model = PretrainedImageToDEM(use_instance_norm=ARGS.use_instancenorm).to(device)

        # Run the model
        dem = model(sat)
        dem = convert_dem_batch(dem)

        # Create a grid of images for visualization
        grid = torchvision.utils.make_grid(
            torch.cat([sat, dem], dim=0),
            normalize=True,
            value_range=(-1, 1),
            nrow=ARGS.batch_size,
        )

        # Save the grid
        torchvision.utils.save_image(grid, "image_to_dem.png")
    elif ARGS.dem_to_image:
        model = PretrainedDEMToImage(use_instance_norm=ARGS.use_instancenorm).to(device)

        # Run the model
        sat = model(gtif)
        gtif = convert_dem_batch(gtif)

        # Create a grid of images for visualization
        grid = torchvision.utils.make_grid(
            torch.cat([gtif, sat], dim=0),
            normalize=True,
            value_range=(-1, 1),
            nrow=ARGS.batch_size,
        )

        # Save the grid
        torchvision.utils.save_image(grid, "dem_to_image.png")
    elif ARGS.progan:
        model = PretrainedProGAN().to(device)

        # Run the model
        random_noise = torch.randn(ARGS.batch_size, 256, 1, 1, device=device)
        sat = model(random_noise)

        # Create a grid of images for visualization
        grid = torchvision.utils.make_grid(
            sat,
            normalize=True,
            value_range=(-1, 1),
        )

        # Save the grid
        torchvision.utils.save_image(grid, "progan.png")
    elif ARGS.inpainting:
        model = PretrainedDeepfillV1().to(device)

        # Run the model
        mask = torch.zeros(sat.size(0), 1, sat.size(2), sat.size(3), device=device)

        # For each sample in the batch, create a random mask
        for i in range(ARGS.batch_size):
            top, left, height, width = random_bbox(
                sat.size(2),
                sat.size(3),
                min_scale_factor=ARGS.min_scale_factor,
                max_scale_factor=ARGS.max_scale_factor,
            )
            mask[i, 0, top : top + height, left : left + width] = 1

        sat = sat.to(device)

        masked_x = sat * (1 - mask)
        _, out_sat = model(masked_x, mask)

        # Create a grid of images for visualization
        grid = torchvision.utils.make_grid(
            torch.cat([sat, masked_x, out_sat], dim=0),
            normalize=True,
            value_range=(-1, 1),
            nrow=ARGS.batch_size,
        )

        # Save the grid
        torchvision.utils.save_image(grid, "inpainting.png")
    else:
        raise ValueError("No model selected")
