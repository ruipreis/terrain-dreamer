from typing import Tuple
import numpy as np
import noise

_POSSIBLE_BIOMES = [
    ["ocean", 0.2],
    ["beach", 0.4],
    ["plains", 0.6],
    ["hills", 0.8],
    ["mountains", 1.0],
]


def biome() -> Tuple[str, float]:
    # The biome is attributed a random betwene the possible biomes choices
    rand_biome = _POSSIBLE_BIOMES[np.random.randint(0, len(_POSSIBLE_BIOMES))]

    return rand_biome[0], rand_biome[1]


def generate_tile(
    tile_x: int,
    tile_y: int,
    tile_size: int,
    scale: float = 350.0,
    octaves: int = 10,
    persistence: float = 0.2,
    lacunarity: float = 3.0,
    base: int = 0,
) -> np.ndarray:
    values = np.zeros((tile_size, tile_size))

    for i in range(tile_size):
        for j in range(tile_size):
            values[i, j] = noise.pnoise2(
                (tile_x + j) / scale,
                (tile_y + i) / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                base=base,
            )

    # Derive the biome from the mean height
    biome_name, biome_height = biome()

    # Rescale according to the biome
    values *= biome_height

    return biome_name, values


if __name__ == "__main__":
    map_tiles = 3
    tile_size = 512

    import matplotlib.pyplot as plt

    for i in range(map_tiles):
        for j in range(map_tiles):
            biome_name, tile = generate_tile(i * tile_size, j * tile_size, tile_size)
            plt.imshow(tile, cmap="gray")
            plt.title("(" + str(j) + "," + str(i) + ")")
            plt.savefig("perlin_noise(" + str(j) + "," + str(i) + ").png")
