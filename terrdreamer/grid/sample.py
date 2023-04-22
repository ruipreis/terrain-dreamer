from terrdreamer.models.pretrained import PretrainedProGAN
from terrdreamer.grid import FeatureVectorExtractor, random_sample
from terrdreamer.grid.run import get_client
import cv2
import numpy as np

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-name", type=str, default="satelite_grids")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-dir", type=Path, default="data/satelite_images")
    args = parser.parse_args()

    collection_name = args.collection_name
    batch_size = args.batch_size

    progan_model = PretrainedProGAN()
    progan_model = progan_model.to("cuda")

    # Load a feature extractor
    feature_extractor = FeatureVectorExtractor()
    feature_extractor = feature_extractor.to("cuda")

    # Instantiate a Qdrant client and recreate the collection
    client = get_client()

    # Generate a batch of random satelite
    rand_sat = progan_model(random_sample(batch_size))

    # The output of the model is in the range [-1, 1], need to convert to [0, 255]
    rand_sat = (rand_sat + 1) * 127.5

    # Extract the feature vectors
    feature_vectors = feature_extractor(rand_sat).detach().cpu()

    # Now find the nearest neighbors
    nearest_neighbors = {}

    for i in range(batch_size):
        search_results = client.search(
            collection_name=collection_name,
            query_vector=feature_vectors[i].tolist(),
            limit=5,
        )

        nearest_neighbors[i] = [search_result.id for search_result in search_results]

    # Load the nearest neighbors and generate a image which shows the original image
    # and the nearest neighbors below it
    for i in range(batch_size):
        # Load the original image
        original_image = (
            rand_sat[i].detach().cpu().numpy().astype("uint8").transpose(1, 2, 0)
        )
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

        # Load the nearest neighbors
        neighbor_images = [
            cv2.imread(
                str(args.image_dir / f"{nearest_neighbor}.png"),
                cv2.IMREAD_COLOR,
            )
            for nearest_neighbor in nearest_neighbors[i]
        ]

        # Resize the images to be the same size
        neighbor_images = [
            cv2.resize(neighbor_image, (256, 256)) for neighbor_image in neighbor_images
        ]

        # Stack the images vertically
        h2_stack = cv2.hconcat(neighbor_images)

        h1_stack = np.zeros_like(h2_stack)
        h1_stack[:256, :256] = original_image
        stacked_image = cv2.vconcat([h1_stack, h2_stack])

        # Save the image
        cv2.imwrite(str(f"neighbors_{i}.jpg"), stacked_image)
