from pathlib import Path
import argparse
import random
from PIL import Image
import rasterio
import numpy as np
import zipfile
import tempfile
import ray
from ray.util.queue import Queue
from tqdm import tqdm

MIN_ELEVATION = -283
MAX_ELEVATION = 7943


def _find_bounds(sat_folder: Path, gtif_folder: Path):
    # Only considers gtif files that have a match sat file
    sat_files = [x.stem for x in list(sat_folder.glob("*.jpg"))]

    # Open all the gTIF files and find the min and max elevation
    min_elevation = float("inf")
    max_elevation = -float("inf")

    for sat_id in tqdm(sat_files):
        gtif_file = gtif_folder / f"{sat_id}.tif"

        with rasterio.open(gtif_file) as src:
            gtif_data = np.array(src.read(1))

            # Only consider if 50% of the pixels are above sea level
            if ((gtif_data <= 0).sum() / gtif_data.size) > 0.5:
                print("Found invalid gTIF file: ", gtif_file)
                continue

            min_elevation = min(min_elevation, gtif_data.min())
            max_elevation = max(max_elevation, gtif_data.max())

    return min_elevation, max_elevation


def prepare_dataset(sat_folder: Path, gtif_folder: Path, out_path: Path):
    def _read_sat_and_gtif(sat_file: Path, gtif_file: Path):
        # For each satellite image, we need to know the width and height
        # we'll keep only the images that are 512x512
        with Image.open(sat_file) as img:
            width, height = img.size

            # Convert the image to numpy
            sat_img = np.array(img)

            if width != 512 or height != 512:
                return

            # Now we read the gTIF file and get the elevation data
            with rasterio.open(gtif_file) as src:
                gtif_data = np.array(src.read(1))

        # Split into 256x256 patches
        for i in range(0, 512, 256):
            for j in range(0, 512, 256):
                sat_patch = sat_img[i : i + 256, j : j + 256]
                gtif_patch = gtif_data[i : i + 256, j : j + 256]

                # Need to validate the gTIF patch before sending to writer thread,
                # to be a part of the dataset atleast 50% of the pixels must be above sea level
                if ((gtif_patch <= 0).sum() / gtif_patch.size) > 0.5:
                    continue

                # Now simply write the patches to the output folder
                out_file = out_path / f"{sat_file.stem}_{i}_{j}.npz"
                np.savez(out_file, SAT=sat_patch, GTIF=gtif_patch)

    # List all the satellite images
    sat_files = list(sat_folder.glob("*.jpg"))

    # Now create pairs between the satellite images and the gTIF files
    sat_gtif_pairs = [
        (sat_file, gtif_folder / f"{sat_file.stem}.tif") for sat_file in sat_files
    ]

    for sat_file, gtif_file in tqdm(sat_gtif_pairs):
        _read_sat_and_gtif(sat_file, gtif_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sat-folder", type=Path, required=True)
    parser.add_argument("--gtif-folder", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--find-bounds", action="store_true")
    args = parser.parse_args()

    if args.find_bounds:
        min_value, max_value = _find_bounds(args.sat_folder, args.gtif_folder)
        print("Min elevation: ", min_value)
        print("Max elevation: ", max_value)
    else:
        prepare_dataset(args.sat_folder, args.gtif_folder, args.out_path)
