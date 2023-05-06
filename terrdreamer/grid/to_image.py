from PIL import Image
import numpy as np
import h5py
import cv2


def to_image(grid, h5_file, tile_size=(256, 256)):
    grid_height, grid_width = grid
    tile_height, tile_width = tile_size

    # Create an empty image with the size of the final image
    result_image = Image.new(
        "RGB", (grid_width * tile_width, grid_height * tile_height)
    )

    for i in range(grid_height):
        for j in range(grid_width):
            tile_key = str(tuple((i, j)))
            tile_image = Image.fromarray(np.array(h5_file[tile_key])[..., ::-1])

            # Paste the tile image into the result image at the correct position
            result_image.paste(tile_image, (j * tile_width, i * tile_height))

    return result_image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="inpainting.h5")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as h5_file:
        cv2.imwrite("output_image.png", h5_file["tiles"][:])
