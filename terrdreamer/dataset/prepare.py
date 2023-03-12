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

def prepare_dataset(sat_folder: Path, gtif_folder: Path, zip_file: Path):
    @ray.remote
    def _read_sat_and_gtif(q:Queue, sat_file: Path, gtif_file: Path):
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

                # We'll save the patches in a dictionary
                # The key will be the patch name
                # The value will be a tuple of the satellite image and the gTIF data
                q.put((f"{sat_file.stem}_{i}_{j}", sat_patch, gtif_patch))

    @ray.remote
    def _write_to_zip(q:Queue, zip_file: Path):
        with zipfile.ZipFile(zip_file, "w") as zip:
            while True:
                x = q.get()

                if x is None:
                    break
                else:
                    patch_name, sat_patch, gtif_patch = x


                info = {
                    "SAT": sat_patch,
                    "GTIF": gtif_patch,
                }

                tmp_npz = tempfile.NamedTemporaryFile(suffix=".npz")
                np.savez(tmp_npz, **info)
                zip.write(tmp_npz.name, arcname=f"{patch_name}.npz")
                tmp_npz.close()

    # List all the satellite images
    sat_files = list(sat_folder.glob("*.jpg"))

    # Now create pairs between the satellite images and the gTIF files
    sat_gtif_pairs = [
        (sat_file, gtif_folder / f"{sat_file.stem}.tif") for sat_file in sat_files
    ]

    q = Queue()

    # Launch a thread to write to the zip file
    write_to_zip = _write_to_zip.remote(q, zip_file)

    # Launch a thread for each satellite image
    write_refs = [_read_sat_and_gtif.remote(q, sat_file, gtif_file) for sat_file, gtif_file in sat_gtif_pairs]

    # Wait for all the threads to finish
    ray.get(write_refs)

    # Signal the writing thread to finish
    q.put(None)

    # Wait for the writing thread to finish
    ray.get(write_to_zip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sat-folder", type=Path, required=True)
    parser.add_argument("--gtif-folder", type=Path, required=True)
    parser.add_argument("--zip-file", type=Path, required=True)
    args = parser.parse_args()

    prepare_dataset(args.sat_folder, args.gtif_folder, args.zip_file)
