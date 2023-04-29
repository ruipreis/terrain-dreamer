import h5py
import numpy as np
import os
import multiprocessing as mp


def h5_file_manager(
    processing_queue, tile_request_queue, dem_queue, file_path="tiles_and_dem.h5"
):
    # Create the HDF5 file
    with h5py.File(file_path, "a") as h5_file:
        # Process incoming tiles, DEM data, and requests
        while True:
            try:
                # Check for incoming tiles
                tile_data = processing_queue.get(block=False)
                if tile_data is not None:
                    tile_id, tile = tile_data

                    # Convert the tuple to a string to be used as a group name
                    group_name = f"tile_{tile_id[0]}_{tile_id[1]}"

                    # Create a new group for the tile if it doesn't exist
                    if group_name not in h5_file:
                        tile_group = h5_file.create_group(group_name)
                        tile_group.create_dataset("texture", data=tile, dtype=np.uint8)
                    else:
                        h5_file[group_name]["texture"][...] = tile

                    h5_file.flush()
            except mp.queues.Empty:
                pass

            try:
                # Check for incoming DEM data
                dem_data = dem_queue.get(block=False)
                if dem_data is not None:
                    tile_id, dem = dem_data

                    # Convert the tuple to a string to be used as a group name
                    group_name = f"tile_{tile_id[0]}_{tile_id[1]}"

                    # Create a new group for the tile if it doesn't exist
                    if group_name not in h5_file:
                        tile_group = h5_file.create_group(group_name)
                        tile_group.create_dataset("dem", data=dem, dtype=np.float32)
                    else:
                        h5_file[group_name]["dem"][...] = dem

                    h5_file.flush()
            except mp.queues.Empty:
                pass

            try:
                # Check for tile and DEM requests
                request = tile_request_queue.get(block=False)
                if request is not None:
                    tile_id, response_queue, dem_request = request
                    group_name = f"tile_{tile_id[0]}_{tile_id[1]}"

                    if group_name in h5_file:
                        tile = h5_file[group_name]["texture"][...]
                        dem_data = (
                            h5_file[group_name]["dem"][...] if dem_request else None
                        )
                        response_queue.put((tile, dem_data))
                    else:
                        response_queue.put((None, None))
            except mp.queues.Empty:
                pass


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
    beta = beta_sinalizer(i, delta)
    upsilon = beta - delta / 2
    return (upsilon - 0.5, upsilon + 0.5)


delta = 0.1
for i in range(-10, 10):
    print(
        i,
        mask_interval(i, delta),
        grid_interval(i, delta),
        inpainting_interval(i, delta),
    )
