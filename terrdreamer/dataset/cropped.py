import torch
import random
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import (
    ToTensor,
    ToPILImage,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from torchvision import transforms
from pathlib import Path
import numpy as np


def masked_random_crop(
    min_factor: float, max_factor: float, img_tensor: torch.Tensor, minimum_mask=0.1
):
    _, height, width = img_tensor.shape
    crop_width = random.randint(int(width * min_factor), int(width * max_factor))
    crop_height = random.randint(int(height * min_factor), int(height * max_factor))
    random_crop = RandomCrop(size=(crop_height, crop_width), pad_if_needed=True)

    top, left, _, _ = random_crop.get_params(img_tensor, (crop_height, crop_width))
    mask = torch.ones(1, *img_tensor.shape[1:]) * (-1 + minimum_mask / 2)
    mask[:, top : top + crop_height, left : left + crop_width] = 1

    img_tensor[:, top : top + crop_height, left : left + crop_width] = 0

    return img_tensor, mask


class CroppedDataset(Dataset):
    def __init__(
        self, path: Path, min_factor=0.3, max_factor=0.6, limit=None, minimum_mask=0.01
    ):
        self._available_files = list(path.glob("*.npz"))

        if limit is not None:
            self._available_files = random.sample(self._available_files, limit)

        self._min_factor = min_factor
        self._max_factor = max_factor
        self._minimum_mask = minimum_mask
        self._transforms = transforms.Compose(
            [
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ]
        )

    def __len__(self):
        return len(self._available_files)

    def __getitem__(self, idx):
        data = np.load(self._available_files[idx])
        sat_img = torch.tensor(data["SAT"], dtype=torch.float32)
        sat_img = sat_img.permute(2, 0, 1)

        sat_img = self._transforms(sat_img)

        # Keep an unmodified copy of the image
        orig_sat_img = sat_img.clone()

        sat_img, sat_mask = masked_random_crop(
            self._min_factor, self._max_factor, sat_img, minimum_mask=self._minimum_mask
        )

        # Normalize the image
        orig_sat_img = (orig_sat_img / 127.5) - 1
        sat_img = (sat_img / 127.5) - 1

        return orig_sat_img, sat_img, sat_mask


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    dataset = CroppedDataset(args.dataset)

    # Sample a batch of random images from the dataset and show the resulting
    # images in a grid
    rand_batch = random.sample(range(len(dataset)), args.batch_size)
    original, masked, mask = zip(*[dataset[i] for i in rand_batch])
    original_grid = torchvision.utils.make_grid(torch.stack(original), normalize=True)
    masked_grid = torchvision.utils.make_grid(torch.stack(masked), normalize=True)
    mask_grid = torchvision.utils.make_grid(torch.stack(mask), normalize=True)

    # Now save the grid to a file
    torchvision.utils.save_image(original_grid, "original.png")
    torchvision.utils.save_image(masked_grid, "masked.png")
    torchvision.utils.save_image(mask_grid, "mask.png")
