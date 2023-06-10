from typing import List

import h5py
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from scipy.spatial import KDTree
from tqdm import tqdm

from terrdreamer.grid.run import get_client


class QdrantWrapper:
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client
        self.sampled_ids = set()

    def search(
        self, collection_name: str, query_vector: List[float], top_k: int
    ) -> List[int]:
        query_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            limit=top_k,
            params={
                "id_only": "true",
            },
        )

        results = [result.id for result in query_result]
        self.sampled_ids.update(results)
        self.delete(collection_name, results)

        return results

    def delete(self, collection_name: str, ids: List[int]):
        self.qdrant_client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=ids,
            ),
        )

    def count_elements(self, collection_name: str) -> int:
        collection_stats = self.qdrant_client.get_collection(
            collection_name=collection_name
        )
        return collection_stats.points_count


def sort_arrays_by_first(arr1, arr2):
    assert len(arr1) == len(arr2), "Both arrays must have the same length."

    # Get the sorted indices of the first array
    sorted_indices = np.argsort(arr1)

    # Sort both arrays using the sorted indices
    arr1_sorted = arr1[sorted_indices]
    arr2_sorted = arr2[sorted_indices]

    return arr1_sorted, arr2_sorted


def generate_grid(height, width, delta, random_ordering):
    grid = np.zeros((height, width))
    tile_index = np.zeros((height, width), dtype=np.int64)
    current_idx = 0

    for i in range(0, height, delta):
        for j in range(0, width, delta):
            grid[i, j] = 1
            tile_index[i, j] = random_ordering[current_idx]
            current_idx += 1

    return grid, tile_index


def weighted_average(
    matrix,
    tile_index_grid,
    tiles,
    real_tiles,
    features,
    num_neighbors,
    out_path: str,
    qdrant_wrapper: QdrantWrapper,
):
    print(
        "There are a total of fake tiles in qdrant:",
        qdrant_wrapper.count_elements("fake_tiles"),
    )

    # Get the indices of non-zero and zero values
    non_zero_indices = np.argwhere(matrix != 0)
    zero_indices = np.argwhere(matrix == 0)

    # Get the non-zero values and create a KDTree for nearest neighbor search
    non_zero_values = tile_index_grid[non_zero_indices[:, 0], non_zero_indices[:, 1]]
    kdtree = KDTree(non_zero_indices)

    with h5py.File(out_path, "w") as h5_file:
        # Calculate weighted average for each zero value
        for zero_idx in tqdm(zero_indices):
            distances, indices = kdtree.query(zero_idx, k=num_neighbors)
            weights = 1 / distances
            normalized_weights = weights / np.sum(weights)

            interest_indexes, interest_weights = sort_arrays_by_first(
                non_zero_values[indices], normalized_weights
            )
            weighted_average = np.average(
                features[interest_indexes], weights=interest_weights, axis=0
            )

            closest_tile = qdrant_wrapper.search(
                collection_name="fake_tiles",
                query_vector=weighted_average.tolist(),
                top_k=1,
            )[0]

            print("Closest tile: {}".format(closest_tile))

            # Retrieve the tile from the fake_tiles dataset
            fake_tile = tiles[closest_tile]

            # Write to h5 file, with the key being the tuple of the zero_idx
            h5_file.create_dataset(str(tuple(zero_idx)), data=fake_tile)

        # Also add the non zero values to the hdf5 file
        for non_zero_idx in non_zero_indices:
            h5_file.create_dataset(
                str(tuple(non_zero_idx)),
                data=real_tiles[tile_index_grid[non_zero_idx[0], non_zero_idx[1]]],
            )

    print("Total distinct sampled tiles: {}".format(len(qdrant_wrapper.sampled_ids)))

    return matrix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-neighbors", type=int, default=3)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--delta", type=int, default=4)
    args = parser.parse_args()

    with h5py.File("real_tiles.h5", "r") as h5_file:
        tiles = h5_file["tiles"]
        features = h5_file["features"]

        with h5py.File("fake_tiles.h5", "r") as fake_tiles_file:
            fake_tiles = fake_tiles_file["tiles"]

            n_real_tiles = tiles.shape[0]

            print("There are {} real tiles".format(n_real_tiles))

            # create a random ordering of these tiles
            random_ordering = np.random.permutation(n_real_tiles)

            # generate a grid
            grid, tile_index = generate_grid(
                args.height, args.width, args.delta, random_ordering
            )

            result = weighted_average(
                grid,
                tile_index,
                fake_tiles,
                tiles,
                features,
                args.num_neighbors,
                "weighted_average.h5",
                QdrantWrapper(get_client()),
            )
            print("\nMatrix after weighted average:")
            print(result)
