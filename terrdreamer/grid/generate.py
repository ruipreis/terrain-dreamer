import logging
import math

import h5py
import numpy as np
from qdrant_client.models import PointStruct
from tqdm import tqdm

from terrdreamer.grid import FeatureVectorExtractor, random_sample
from terrdreamer.grid.run import get_client, recreate_collection
from terrdreamer.models.pretrained import PretrainedProGAN

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(process)d] %(levelname)s %(message)s"
)
logger = logging.getLogger("httpx")
logger.setLevel(logging.WARNING)
logger = logging.getLogger("httpcore")
logger.setLevel(logging.WARNING)


def get_tile_manager_context(collection_name: str = "fake-tiles", device: str = "cuda"):
    progan_model = PretrainedProGAN()
    progan_model = progan_model.to(device)

    # Load a feature extractor
    feature_extractor = FeatureVectorExtractor()
    feature_extractor = feature_extractor.to(device)

    # Instantiate a Qdrant client and recreate the collection
    if collection_name is not None:
        qdrant_client = get_client()
        recreate_collection(qdrant_client, collection_name, feature_extractor.get_dim())
    else:
        qdrant_client = None

    return progan_model, feature_extractor, qdrant_client


def get_random_features(
    batch_size: int,
    progan_model: PretrainedProGAN,
    feature_extractor: FeatureVectorExtractor,
    device: str = "cuda",
):
    # Generate a batch of random satelite
    rand_sat = progan_model(random_sample(batch_size, device=device))

    # The output of the model is in the range [-1, 1], need to convert to [0, 255]
    rand_sat = (rand_sat + 1) * 127.5

    # Extract the feature vectors
    feature_vectors = feature_extractor(rand_sat).detach().cpu()
    rand_sat = rand_sat.detach().cpu().numpy()

    # Transpose the random sat images to be in the format (batch_size, H, W, 3)
    rand_sat = np.transpose(rand_sat, (0, 2, 3, 1))

    # Flip the last dimension to be in BGR format
    rand_sat = rand_sat.astype(np.uint8)

    return rand_sat, feature_vectors


# Calculates the amount of real and fake tiles a grid is expected to have
def get_num_real_fake_tiles(H: int, W: int, delta: int):
    relevant_rows = (H + delta - 1) // delta
    relevant_cols = (W + delta - 1) // delta
    num_real_tiles = relevant_rows * relevant_cols

    # Number of fake tiles
    num_fake_tiles = H * W - num_real_tiles

    return num_real_tiles, num_fake_tiles


# Each pool worker will have its own context and will be responsible for
# generating the required amount of tiles
def generate_saveable_samples(
    h5_file_path: str,
    progan: PretrainedProGAN,
    feature_extractor: FeatureVectorExtractor,
    qdrant_client,
    num_tiles: int,
    device: str,
    batch_size: int,
    collection_name: str,
    type: str = "fake",
):
    with h5py.File(h5_file_path, "w") as h5_file:
        logging.debug(f"Creating a dataset for the {type} tiles")
        # Create a dataset for the fake tiles
        tiles = h5_file.create_dataset(
            "tiles",
            shape=(num_tiles, 256, 256, 3),
            dtype=np.uint8,
        )

        if type == "real":
            features = h5_file.create_dataset(
                "features",
                shape=(num_tiles, feature_extractor.get_dim()),
                dtype=np.float32,
            )
        else:
            features = None

        logging.debug(f"Finished creating the {type} tiles dataset")

        # Fill the fake tiles dataset
        for i in tqdm(range(0, num_tiles, batch_size)):
            real_end = min(i + batch_size, num_tiles)

            rand_sat, feature_vectors = get_random_features(
                real_end - i, progan, feature_extractor, device=device
            )

            # Add the fake tiles to the dataset
            tiles[i : i + (real_end - i)] = rand_sat

            if features is not None:
                features[i : i + (real_end - i)] = feature_vectors

            # Add the fake tiles to the collection
            if qdrant_client is not None:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=i + j,
                            vector=feature_vectors[j].tolist(),
                        )
                        for j in range(real_end - i)
                    ],
                )

        logging.debug(f"Finished generating {type} tiles")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--delta", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--real-h5-file", type=str, default="real_tiles.h5")
    parser.add_argument("--fake-h5-file", type=str, default="fake_tiles.h5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--collection-name", type=str, default="fake_tiles")
    args = parser.parse_args()

    logging.debug("Finished parsing arguments: " + str(vars(args)))

    # Calculate the number of real and fake tiles
    num_real_tiles, num_fake_tiles = get_num_real_fake_tiles(
        args.height, args.width, args.delta
    )
    logging.info(f"Number of real tiles: {num_real_tiles}")
    logging.info(f"Number of fake tiles: {num_fake_tiles}")

    logging.debug("Starting tile manager context")
    progan, feature_extractor, qdrant_client = get_tile_manager_context(
        collection_name=args.collection_name, device=args.device
    )

    generate_saveable_samples(
        h5_file_path=args.real_h5_file,
        progan=progan,
        feature_extractor=feature_extractor,
        qdrant_client=None,
        num_tiles=num_real_tiles,
        device=args.device,
        batch_size=args.batch_size,
        collection_name=None,
        type="real",
    )

    generate_saveable_samples(
        h5_file_path=args.fake_h5_file,
        progan=progan,
        feature_extractor=feature_extractor,
        qdrant_client=qdrant_client,
        num_tiles=num_fake_tiles,
        device=args.device,
        batch_size=args.batch_size,
        collection_name=args.collection_name,
        type="fake",
    )
