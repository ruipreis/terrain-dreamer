import torch
import random
from torchvision.transforms import RandomCrop
from typing import Tuple
from pathlib import Path
from terrdreamer.dataset import AW3D30Dataset
from terrdreamer.models.infinity_grid import DeepFillV1


def local_patch(x, bbox):
    """
    Crop local patch according to bbox.
    Args:
        x: input
        bbox: (top, left, height, width)
    Returns:
        torch.Tensor: local patch
    """
    top, left, height, width = bbox
    patch = x[:, :, top : top + height, left : left + width]
    return patch


def random_bbox(
    height, width, min_factor: float, max_factor: float
) -> Tuple[int, int, int, int]:
    img_tensor = torch.ones(1, height, width)
    crop_height = random.randint(int(height * min_factor), int(height * max_factor))
    crop_width = random.randint(int(width * min_factor), int(width * max_factor))
    random_crop = RandomCrop(size=(crop_height, crop_width), pad_if_needed=True)
    top, left, _, _ = random_crop.get_params(img_tensor, (crop_height, crop_width))
    return top, left, crop_height, crop_width


def train(
    train_dataset: Path,
    limit: int = 1000,
    batch_size: int = 16,
    pretrained_generator_path: Path = None,
    pretrained_local_discriminator_path: Path = None,
    pretrained_global_discriminator_path: Path = None,
):
    # Instantiate the dataset
    dataset = AW3D30Dataset(train_dataset, limit=limit)

    # Get the matching loader for the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    deepfill_model = DeepFillV1()

    # If needed, load pretrained weights
    deepfill_model.load_pretrained_if_needed(
        pretrained_generator_path,
        pretrained_local_discriminator_path,
        pretrained_global_discriminator_path,
    )

    sat, _ = dataset[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train(args.train_dataset, limit=args.limit, batch_size=args.batch_size)
