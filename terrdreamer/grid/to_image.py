from PIL import Image
import numpy as np
import h5py
import cv2

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="inpainting.h5")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as h5_file:
        max_height = h5_file["max_height"][()]
        max_width = h5_file["max_width"][()]
        cv2.imwrite(
            "output_image2.png",
            h5_file["tiles"][:max_height, :max_width][:][..., ::-1],
        )
        cv2.imwrite("masks.png", h5_file["masks"][:] * 255)
