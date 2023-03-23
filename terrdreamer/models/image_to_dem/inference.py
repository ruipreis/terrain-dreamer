import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from terrdreamer.dataset import tiff_to_jpg
from terrdreamer.models.image_to_dem import DEM_Pix2Pix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-path", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Load generator - this is the model that we will use to generate the DEM
    pix2pix = DEM_Pix2Pix(3, 1, inference=True)
    pix2pix.load_generator(args.generator_path)
    pix2pix.generator.to("cuda")
    pix2pix.generator.set_requires_grad(False)

    # Load the image and convert it to a tensor
    img = Image.open(args.rgb)
    img = img.resize((256, 256))
    img = np.array(img)
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to("cuda")
    img_tensor = (img_tensor / 127.5) - 1

    # Generate the DEM
    dem = pix2pix.generator(img_tensor)

    # Save the DEM
    dem = dem[0].detach().cpu()

    # Now save the image
    tiff_to_jpg(dem, convert=True, out_path=args.output)
