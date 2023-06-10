import argparse

import h5py
import numpy as np
import torch
from tqdm import tqdm

from terrdreamer.models.pretrained import PretrainedImageToDEM, convert_dem_batch


def vector_linspace(v1, v2, num_points=10):
    """
    Generate linearly spaced vectors between two N-dimensional vectors.

    Args:
        v1 (np.array): The start N-dimensional vector.
        v2 (np.array): The end N-dimensional vector.
        num_points (int): The number of points to generate between v1 and v2, inclusive.

    Returns:
        np.array: An array of linearly spaced N-dimensional vectors.
    """
    # Create linearly spaced arrays for each component of the vectors
    component_arrays = [np.linspace(v1[i], v2[i], num_points) for i in range(len(v1))]

    # Combine the component arrays into an array of vectors
    vector_array = np.column_stack(tuple(component_arrays))

    # Swap the first and second axis
    vector_array = np.swapaxes(vector_array, 0, 1)

    return vector_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="inpainting.h5")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--delta", type=int, default=10)
    args = parser.parse_args()

    # Instantiate the model for DEM estimation
    dem_model = PretrainedImageToDEM(use_instance_norm=True).to(args.device)

    with h5py.File(args.file, "a") as h5_file:
        tiles = h5_file["tiles"]
        height, width = tiles.shape[:2]

        # Create a dataset for the DEMs if it doesn't exist
        if "dems" in h5_file:
            del h5_file["dems"]

        dems = h5_file.create_dataset(
            "dems",
            shape=(height, width, 1),
            dtype="float32",
            chunks=(args.tile_size, args.tile_size, 1),
        )

        # Keep a batch of tiles to send to the model
        batch = []
        batch_indices = []

        # Iterate over the tiles and estimate the DEM
        for i in tqdm(range(0, height, args.tile_size)):
            for j in range(0, width, args.tile_size):
                # Get the tile
                tile = tiles[i : i + args.tile_size, j : j + args.tile_size]
                tile = tile.astype("float32")

                # Normalize the tile
                tile = torch.from_numpy(tile / 127.5 - 1).permute(2, 0, 1)

                # Add the tile to the batch
                batch.append(tile)
                batch_indices.append((i, j))

                if len(batch) == args.batch_size:
                    # Run the model
                    tile_batch = torch.stack(batch).to(args.device)
                    dem = dem_model(tile_batch)
                    dem = convert_dem_batch(dem, repeat=False)

                    # Convert the DEM to numpy
                    dem = dem.cpu().numpy()
                    dem = dem[:, 0, :, :, None]

                    # Add the DEM to the dataset
                    for single_dem, (i, j) in zip(dem, batch_indices):
                        dems[
                            i : i + args.tile_size, j : j + args.tile_size
                        ] = single_dem

                    # Clear the batch
                    batch = []
                    batch_indices = []

        # Run the model for the remaining tiles
        if len(batch) > 0:
            # Run the model
            tile_batch = torch.stack(batch).to(args.device)
            dem = dem_model(tile_batch)
            dem = convert_dem_batch(dem, repeat=False)

            # Convert the DEM to numpy
            dem = dem.cpu().numpy()
            dem = dem[:, 0, :, :, None]

            # Add the DEM to the dataset
            for single_dem, (i, j) in zip(dem, batch_indices):
                dems[i : i + args.tile_size, j : j + args.tile_size] = single_dem

        # Interpolate between neighbour DEMs to fill the gaps
        for i in range(0, height, args.tile_size):
            for j in range(args.tile_size, width, args.tile_size):
                dem1 = dems[i : i + args.tile_size, [j - args.delta, j + args.delta], 0]

                # Fill the gap with the linspace between the two DEMs
                dems[
                    i : i + args.tile_size, (j - args.delta) : (j + args.delta)
                ] = vector_linspace(dem1[..., 0], dem1[..., 1], 2 * args.delta)[
                    ..., None
                ]

        for i in range(args.tile_size, height, args.tile_size):
            for j in range(0, width, args.tile_size):
                dem1 = dems[[i - args.delta, i + args.delta], j : j + args.tile_size, 0]

                # Fill the gap with the linspace between the two DEMs
                dems[
                    (i - args.delta) : (i + args.delta), j : j + args.tile_size
                ] = np.swapaxes(
                    vector_linspace(dem1[0], dem1[1], 2 * args.delta), 0, 1
                )[
                    ..., None
                ]
