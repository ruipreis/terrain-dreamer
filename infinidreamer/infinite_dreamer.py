import cv2
import argparse
import torch
import numpy as np

from satellite_image_generator import SatelliteImageGenerator
from image_cluster import ImageCluster
from grid_map import GridMap
from generative_models import PretrainedProGAN, PretrainedDeepFillV1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a grid map using satellite images"
    )
    parser.add_argument(
        "--num-images", type=int, default=4, help="Number of images to generate"
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.6,
        help="How similar the generated images should be",
    )
    parser.add_argument(
        "--grid-size", type=int, default=2, help="Size of the grid (square grid)"
    )
    parser.add_argument(
        "--spacing", type=int, default=64, help="Spacing between images in the grid"
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=4,
        help="Number of clusters for image clustering",
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Size of the generated images"
    )
    parser.add_argument(
        "--num-outputs", type=int, default=10, help="Number of output files to generate"
    )
    return parser.parse_args()


def generate_satellite_images(device, num_images, similarity_weight, diversity_weight):
    """Generate satellite images using PretrainedProGAN."""
    progan_model = PretrainedProGAN().to(device)
    satellite_image_generator = SatelliteImageGenerator(
        progan_model=progan_model, device=device
    )
    return satellite_image_generator.generate_images_with_features(
        batch_size=num_images,
        similarity_weight=similarity_weight,
        diversity_weight=diversity_weight,
    )


def cluster_images(image_vector_pairs, max_clusters):
    """Cluster images based on their feature vectors."""
    image_cluster = ImageCluster(max_clusters=max_clusters)
    return image_cluster.cluster_images(image_feature_vector_pairs=image_vector_pairs)


def create_map(device, grid_size, spacing, clustered_images):
    """Create a grid map and inpaint gaps using DeepFillV1."""
    deepfill_model = PretrainedDeepFillV1().to(device)
    grid_map_generator = GridMap(
        grid_size=grid_size,
        spacing=spacing,
        clustered_images=clustered_images,
        deepfill_model=deepfill_model,
        device=device,
    )
    grid_map, mask = grid_map_generator.create_map()
    return grid_map_generator, grid_map, mask


def apply_border_inpainting(grid_map_generator, grid_map, mask):
    """Apply border inpainting to a grid map."""
    grid_map_copy = grid_map.copy()
    return grid_map_generator.apply_border_inpainting(grid_map=grid_map_copy, mask=mask)


def apply_random_inpainting_in_windows(grid_map_generator, no_borders_grid_map):
    """Apply random inpainting to a grid map."""
    no_borders_grid_map_copy = no_borders_grid_map.copy()
    return grid_map_generator.apply_random_inpainting_in_windows(
        grid_map=no_borders_grid_map_copy
    )


def save_images(output_file, images):
    """Save images to disk."""
    for i, image in enumerate(images):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_file}_{i}.png", image_bgr)


def main(args, output_file):
    """Generate a grid map using satellite images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_vector_pairs = generate_satellite_images(
        device=device,
        num_images=args.num_images,
        similarity_weight=args.similarity,
        diversity_weight=(1 - args.similarity),
    )

    clustered_images = cluster_images(
        image_vector_pairs=image_vector_pairs, max_clusters=args.num_clusters
    )

    grid_map_generator, grid_map, mask = create_map(
        device=device,
        grid_size=args.grid_size,
        spacing=args.spacing,
        clustered_images=clustered_images,
    )

    save_images(output_file=f"no_inpainting_{output_file}", images=[grid_map])

    no_borders_grid_map = apply_border_inpainting(
        grid_map_generator=grid_map_generator, grid_map=grid_map, mask=mask
    )

    save_images(
        output_file=f"border_inpainting_{output_file}", images=[no_borders_grid_map]
    )

    random_inpainting_grid_map = apply_random_inpainting_in_windows(
        grid_map_generator=grid_map_generator, no_borders_grid_map=no_borders_grid_map
    )

    save_images(
        output_file=f"final_inpainting_{output_file}",
        images=[random_inpainting_grid_map],
    )

    space = np.zeros((grid_map.shape[0], 15, 3), dtype=np.uint8)

    combined_image = np.hstack(
        (grid_map, space, no_borders_grid_map, space, random_inpainting_grid_map)
    )

    combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"combined_{output_file}.png", combined_image_bgr)


if __name__ == "__main__":
    args = parse_args()

    for i in range(args.num_outputs):
        output_file = f"grid_map_{i}"

        main(args, output_file)
