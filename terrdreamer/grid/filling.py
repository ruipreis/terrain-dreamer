import random
import h5py
import math
import numpy as np
from terrdreamer.models.pretrained import PretrainedDeepfillV1
from tqdm import tqdm
import torch


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
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--inter-real-tile-spacing", type=int, default=3)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grid-file", type=str, default="weighted_average.h5")
    parser.add_argument("--out-grid-file", type=str, default="inpainting.h5")
    parser.add_argument("--batch-size", type=int, default=16)

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
    real_batch_size = min(args.batch_size, args.width)

    # Now, we can to perform inpainting on all of the possible inpaiting intervals
    # in the grid.
    inpainting_intervals = [
        (inpainting_interval(i, args.delta), inpainting_interval(j, args.delta))
        for i in range(args.height)
        for j in range(args.width)
    ]

    # Each inpaint interval corresponds to a tile in the grid. thus, we also want to generate
    # random inpaintings for each tile. A later script will take the box and convert into a
    # valid crop to inpaint in. We only want to inpaint in the regions that are not masked.
    delta_adjustment_constant = 0.5 - args.delta / 2

    # Instantiate a model to perform inpainting on the tiles
    model = PretrainedDeepfillV1().to(args.device)

    random_inpaintings = [
        [
            (
                # Adjust to delta
                random.uniform(
                    vh[0] + delta_adjustment_constant, vh[1] - delta_adjustment_constant
                ),
                random.uniform(
                    vw[0] + delta_adjustment_constant, vw[1] - delta_adjustment_constant
                ),
            )
            for _ in range(args.n_random_inpaintings)
        ]
        for vh, vw in inpainting_intervals
    ]

    print(f"Delta adjusted height: {delta_adjusted_height}")
    print(f"Delta adjusted width: {delta_adjusted_width}")

    print("Creating delta adjusted grid")

    with h5py.File(args.out_grid_file, "w") as h5_file:
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

        #     h5_file[inpainting_key] = n\p.array([vh, vw])
