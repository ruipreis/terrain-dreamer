from PIL import Image
import numpy as np
import h5py


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


with h5py.File("weighted_average.h5", "r") as h5_file:
    image = to_image((20, 20), h5_file)
    image.save("output_image.png")
