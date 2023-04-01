import torch
import random
from torchvision.transforms import RandomCrop
from typing import Tuple


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
