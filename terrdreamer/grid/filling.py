import math
import random

import h5py
import numpy as np
import torch
from tqdm import tqdm

from terrdreamer.models.pretrained import PretrainedDeepfillV1


def random_crop_with_point(image_shape, crop_size, point):
    """
    This function generates a bounding box with the specified crop size containing the given point.
    The point will be centered in the bounding box when possible, while preserving the image shape dimensions.

    Args:
        image_shape (tuple): The shape of the reference frame (height, width)
        crop_size (tuple): The desired size of the crop (crop_height, crop_width)
        point (tuple): The point to be included in the crop (y, x)

    Returns:
        tuple: The bounding box coordinates in the format (y_min, x_min, y_max, x_max)
    """

    image_height, image_width = image_shape
    crop_height, crop_width = crop_size
    y, x = point

    # Calculate the minimum and maximum possible coordinates for the point
    y_min = max(0, y - crop_height // 2)
    y_max = min(image_height - crop_height, y - crop_height // 2)
    x_min = max(0, x - crop_width // 2)
    x_max = min(image_width - crop_width, x - crop_width // 2)

    # Clip the coordinates to ensure they are within the image boundaries
    y_min = np.clip(y_min, 0, image_height - crop_height)
    y_max = np.clip(y_max, 0, image_height - crop_height)
    x_min = np.clip(x_min, 0, image_width - crop_width)
    x_max = np.clip(x_max, 0, image_width - crop_width)

    return (y_min, x_min, y_min + crop_height, x_min + crop_width)


def random_crop(
    tile_size,
    min_scale_factor: float = 0.3,
    max_scale_factor: float = 0.5,
):
    crop_height = random.randint(
        int(tile_size * min_scale_factor), int(tile_size * max_scale_factor)
    )
    crop_width = random.randint(
        int(tile_size * min_scale_factor), int(tile_size * max_scale_factor)
    )
    return crop_height, crop_width


def square_bounding_box(rectangular_crop, fixed_square_size, image_shape):
    """
    This function generates a square bounding box that contains the given rectangular crop.
    The shape of the square bounding box is constant and provided as an input parameter.
    All values in the bounding box will be positive.

    Args:
        rectangular_crop (tuple): The rectangular crop bounding box (y_min, x_min, y_max, x_max)
        fixed_square_size (int): The fixed size of the square bounding box
        image_shape (tuple): The shape of the reference frame (height, width)

    Returns:
        tuple: The square bounding box coordinates in the format (y_min, x_min, y_max, x_max)
    """

    y_min, x_min, y_max, x_max = rectangular_crop
    crop_height = y_max - y_min
    crop_width = x_max - x_min
    image_height, image_width = image_shape

    # Calculate the amount to expand each dimension to achieve the fixed square size
    expand_y = (fixed_square_size - crop_height) // 2
    expand_x = (fixed_square_size - crop_width) // 2

    # Calculate the square bounding box coordinates and ensure they are positive
    square_y_min = max(0, y_min - expand_y)
    square_x_min = max(0, x_min - expand_x)
    square_y_max = min(image_height, y_max + expand_y)
    square_x_max = min(image_width, x_max + expand_x)

    # Adjust the square bounding box size if necessary due to clipping
    if square_y_max - square_y_min < fixed_square_size:
        if square_y_min > 0:
            square_y_min -= fixed_square_size - (square_y_max - square_y_min)
        else:
            square_y_max += fixed_square_size - (square_y_max - square_y_min)

    if square_x_max - square_x_min < fixed_square_size:
        if square_x_min > 0:
            square_x_min -= fixed_square_size - (square_x_max - square_x_min)
        else:
            square_x_max += fixed_square_size - (square_x_max - square_x_min)

    return (square_y_min, square_x_min, square_y_max, square_x_max)


def beta_sinalizer(i, delta: float):
    return i * (1 + delta)


def mask_interval(i, delta):
    beta = beta_sinalizer(i, delta)
    return (beta - delta, beta)


def grid_interval(i, delta):
    beta1 = beta_sinalizer(i, delta)
    beta2 = beta_sinalizer(i + 1, delta)
    return (beta1, beta2 - delta)


def inpainting_interval(i, delta):
    grid_i = grid_interval(i, delta)
    grid_next_i = grid_interval(i + 1, delta)
    slice_moment = (grid_i[1] + grid_next_i[0]) / 2
    return (slice_moment - 0.5, slice_moment + 0.5)


def process_batch(model, batch_tiles, batch_masks, device):
    # Perform inpainting on the batch
    batch_tiles = np.stack(batch_tiles)
    batch_masks = np.stack(batch_masks)
    batch_tiles = torch.from_numpy(batch_tiles).to(device)

    # Normalize the tiles
    batch_tiles = batch_tiles.float() / 127.5 - 1

    batch_masks = torch.from_numpy(batch_masks).to(device)
    batch_masks = (batch_masks == 0).float().unsqueeze(-1)

    # Apply mask to the tiles
    batch_tiles = batch_tiles * (1 - batch_masks)
    src_batch_tiles = batch_tiles.cpu().numpy()
    src_batch_tiles = (src_batch_tiles + 1) * 127.5

    # Place in the format (batch_size, 3, H, W)
    batch_tiles = batch_tiles.permute(0, 3, 1, 2)
    batch_masks = batch_masks.permute(0, 3, 1, 2)

    # Perform inpainting
    inpaint_sat = model(batch_tiles, batch_masks)[1].detach().cpu().numpy()

    # Denormalize the inpainted tiles
    inpaint_sat = (inpaint_sat + 1) * 127.5

    # Transpose the inpainted tiles to be in the format (batch_size, H, W, 3)
    inpaint_sat = np.transpose(inpaint_sat, (0, 2, 3, 1))

    # Convert to uint8
    inpaint_sat = inpaint_sat.astype(np.uint8)

    return inpaint_sat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--inter-real-tile-spacing", type=int, default=4)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grid-file", type=str, default="weighted_average.h5")
    parser.add_argument("--out-grid-file", type=str, default="inpainting.h5")
    parser.add_argument("--batch-size", type=int, default=24)

    # The number of random inpaintings to perform on each inpainted tile
    parser.add_argument("--n-random-inpaintings", type=int, default=10)

    args = parser.parse_args()

    # Delta corresponds to the percentage of spacing between tiles, as measure
    # of the proportion of the tile size.

    # For example, if delta = 0.5, then the spacing between tiles is 50% of the
    # tile size.

    # Height and width both represent the size of the grid in tiles in that
    # dimension.

    # Thus, depending on the delta, the total size of the grid in pixels will
    # vary.
    delta_adjusted_height = grid_interval(args.height, args.delta)[1]
    delta_adjusted_width = grid_interval(args.width, args.delta)[1]

    # The real batch size depends on the width
    real_batch_size = args.batch_size

    # Now, we can to perform inpainting on all of the possible inpaiting intervals
    # in the grid.
    inpainting_intervals = [
        (grid_interval(i, args.delta), inpainting_interval(j, args.delta))
        for i in range(args.height)
        for j in range(args.width)
    ]

    for i in range(args.height):
        for j in range(args.width):
            inpainting_intervals.append(
                (inpainting_interval(i, args.delta), grid_interval(j, args.delta))
            )

    for i in range(args.height):
        for j in range(args.width):
            inpainting_intervals.append(
                (inpainting_interval(i, args.delta), inpainting_interval(j, args.delta))
            )

    # Each inpaint interval corresponds to a tile in the grid. thus, we also want to generate
    # random inpaintings for each tile. A later script will take the box and convert into a
    # valid crop to inpaint in. We only want to inpaint in the regions that are not masked.
    delta_adjustment_constant = 0.5 - args.delta / 2

    # Instantiate a model to perform inpainting on the tiles
    model = PretrainedDeepfillV1().to(args.device)

    random_inpaintings = [
        (
            # Adjust to delta
            int(random.uniform(vh[0], vh[1]) * args.tile_size),
            int(random.uniform(vw[0], vw[1]) * args.tile_size),
        )
        for _ in range(args.n_random_inpaintings)
        for vh, vw in inpainting_intervals
    ]

    # For each random inpainting, get a random crop
    image_shape = (
        math.ceil(delta_adjusted_height) * args.tile_size,
        math.ceil(delta_adjusted_width) * args.tile_size,
    )
    random_crops = [
        random_crop_with_point(
            image_shape,
            random_crop(args.tile_size),
            p,
        )
        for p in random_inpaintings
    ]

    random_crops = [
        (square_bounding_box(crop, args.tile_size, image_shape), crop)
        for crop in random_crops
    ]

    print(f"Delta adjusted height: {delta_adjusted_height}")
    print(f"Delta adjusted width: {delta_adjusted_width}")

    print("Creating delta adjusted grid")

    with h5py.File(args.out_grid_file, "w") as h5_file:
        h5_file["max_height"] = int(
            grid_interval(args.height - 1, args.delta)[1] * args.tile_size
        )
        h5_file["max_width"] = int(
            grid_interval(args.width - 1, args.delta)[1] * args.tile_size
        )
        tiles = h5_file.create_dataset(
            "tiles",
            shape=(
                math.ceil(delta_adjusted_height) * args.tile_size,
                math.ceil(delta_adjusted_width) * args.tile_size,
                3,
            ),
            dtype=np.uint8,
        )
        masks = h5_file.create_dataset(
            "masks",
            shape=(
                math.ceil(delta_adjusted_height) * args.tile_size,
                math.ceil(delta_adjusted_width) * args.tile_size,
            ),
            dtype=np.uint8,
        )

        with h5py.File(args.grid_file, "r") as grid_file:
            for i in range(args.height):
                for j in range(args.width):
                    H_grid_interval = grid_interval(i, args.delta)
                    W_grid_interval = grid_interval(j, args.delta)

                    # Adjust to the matching tile size
                    adjusted_H_grid_interval = (
                        int(H_grid_interval[0] * args.tile_size),
                        int(H_grid_interval[1] * args.tile_size),
                    )
                    adjusted_W_grid_interval = (
                        int(W_grid_interval[0] * args.tile_size),
                        int(W_grid_interval[1] * args.tile_size),
                    )

                    tile_key = str(tuple((i, j)))
                    tiles[
                        adjusted_H_grid_interval[0] : adjusted_H_grid_interval[1],
                        adjusted_W_grid_interval[0] : adjusted_W_grid_interval[1],
                    ] = grid_file[tile_key]
                    masks[
                        adjusted_H_grid_interval[0] : adjusted_H_grid_interval[1],
                        adjusted_W_grid_interval[0] : adjusted_W_grid_interval[1],
                    ] = 1

        # Now apply the inpainting intervals
        batch_tiles = []
        batch_masks = []
        batch_regions = []

        for vh, vw in tqdm(inpainting_intervals):
            adjusted_vh = (
                int(vh[0] * args.tile_size),
                int(vh[1] * args.tile_size),
            )
            adjusted_vw = (
                int(vw[0] * args.tile_size),
                int(vw[1] * args.tile_size),
            )

            # Grab the matching tiles and masks
            extracted_tiles = tiles[
                adjusted_vh[0] : adjusted_vh[1], adjusted_vw[0] : adjusted_vw[1]
            ]

            batch_tiles.append(extracted_tiles)
            batch_masks.append(
                masks[adjusted_vh[0] : adjusted_vh[1], adjusted_vw[0] : adjusted_vw[1]]
            )
            batch_regions.append((adjusted_vh, adjusted_vw))

            if len(batch_tiles) == real_batch_size:
                out_sat = process_batch(
                    model,
                    batch_tiles,
                    batch_masks,
                    args.device,
                )

                for i, v in enumerate(batch_regions):
                    tiles[
                        v[0][0] : v[0][1],
                        v[1][0] : v[1][1],
                    ] = out_sat[i]
                    masks[
                        v[0][0] : v[0][1],
                        v[1][0] : v[1][1],
                    ] = 1

                # Reset the batch
                batch_tiles = []
                batch_masks = []
                batch_regions = []

        if len(batch_tiles) > 0:
            out_sat = process_batch(
                model,
                batch_tiles,
                batch_masks,
                args.device,
            )

            for i, v in enumerate(batch_regions):
                tiles[
                    v[0][0] : v[0][1],
                    v[1][0] : v[1][1],
                ] = out_sat[i]
                masks[
                    v[0][0] : v[0][1],
                    v[1][0] : v[1][1],
                ] = 1

        #     h5_file[inpainting_key] = n\p.array([vh, vw])
        batch_tiles = []
        batch_masks = []
        batch_regions = []

        # Apply the random inpaintings
        for window_crop, basis_crop in tqdm(random_crops):
            W_y_min, W_x_min, W_y_max, W_x_max = window_crop
            B_y_min, B_x_min, B_y_max, B_x_max = basis_crop

            # Normalize the basis crop to the window crop
            B_y_min = B_y_min - W_y_min
            B_x_min = B_x_min - W_x_min
            B_y_max = B_y_max - W_y_min
            B_x_max = B_x_max - W_x_min

            # Get the tiles and masks
            batch_tiles.append(
                tiles[
                    W_y_min:W_y_max,
                    W_x_min:W_x_max,
                ]
            )
            box = np.ones((args.tile_size, args.tile_size))
            box[B_y_min:B_y_max, B_x_min:B_x_max] = 0

            batch_masks.append(box)
            batch_regions.append(window_crop)

            if len(batch_tiles) == real_batch_size:
                out_sat = process_batch(
                    model,
                    batch_tiles,
                    batch_masks,
                    args.device,
                )

                for i, v in enumerate(batch_regions):
                    tiles[
                        v[0] : v[2],
                        v[1] : v[3],
                    ] = out_sat[i]

                # Reset the batch
                batch_tiles = []
                batch_masks = []
                batch_regions = []

        if len(batch_tiles) > 0:
            out_sat = process_batch(
                model,
                batch_tiles,
                batch_masks,
                args.device,
            )

            for i, v in enumerate(batch_regions):
                tiles[
                    v[0] : v[2],
                    v[1] : v[3],
                ] = out_sat[i]
