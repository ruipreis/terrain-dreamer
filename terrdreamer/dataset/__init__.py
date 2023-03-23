import random
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch

DEM_MIN_ELEVATION = -82
DEM_MAX_ELEVATION = 7219


def tiff_to_jpg(tiff_data, convert: bool = False, out_path=None):
    # Unnormalize the gtif
    tiff_data = (tiff_data + 1) / 2
    tiff_data = tiff_data * (DEM_MAX_ELEVATION - DEM_MIN_ELEVATION) + DEM_MIN_ELEVATION

    # Normalize the tiff data based on it's min and max, this helps with visualization
    min_value = tiff_data.min()
    max_value = tiff_data.max()
    tiff_data = (tiff_data - min_value) / (max_value - min_value)

    img = tiff_data.squeeze().numpy()

    img = (img * 255).astype(np.uint8)

    img = Image.fromarray(img)
    img = img.convert("L")

    if convert:
        img.save(out_path)

    return img


class AW3D30Dataset(Dataset):
    def __init__(
        self,
        path: Path,
        limit=None,
        normalize: bool = True,
        swap: bool = False,
        for_progan=False,
        new_size=None,
    ):
        self._available_files = list(path.glob("*.npz"))

        if limit is not None:
            self._available_files = random.sample(self._available_files, limit)

        self._normalize = normalize
        self._swap = swap
        self._forprogan = for_progan
        self._newsize = new_size

        if self._newsize is not None:
            self._transform = self._derive_transform(self._newsize)
            self._gtif_transform = self._derive_gtif_transform(self._newsize)

    def _derive_transform(self, newsize):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(newsize),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        return transform

    def _derive_gtif_transform(self, newsize):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(newsize),
            ]
        )

        return transform

    @property
    def newsize(self):
        return self._newsize

    @newsize.setter
    def newsize(self, newsize):
        self._newsize = newsize
        self._transform = self._derive_transform(self._newsize)
        self._gtif_transform = self._derive_gtif_transform(self._newsize)

    def __len__(self):
        return len(self._available_files)

    def __getitem__(self, idx):
        data = np.load(self._available_files[idx])

        if self._forprogan:
            gtif = torch.tensor(data["GTIF"], dtype=torch.float32)

            sat = self._transform(data["SAT"])
            gtif = self._gtif_transform(data["GTIF"])

            # Normalize the inputs
            gtif = (gtif - DEM_MIN_ELEVATION) / (DEM_MAX_ELEVATION - DEM_MIN_ELEVATION)
            gtif = gtif * 2 - 1
        else:
            gtif = torch.tensor(data["GTIF"], dtype=torch.float32)
            sat = torch.tensor(data["SAT"], dtype=torch.float32)

            # Make sure gtif and sat is shaped correctly
            gtif = gtif.unsqueeze(0)

            sat = sat.permute(2, 0, 1)

            if self._normalize:
                sat = (sat / 127.5) - 1
                gtif = (gtif - DEM_MIN_ELEVATION) / (
                    DEM_MAX_ELEVATION - DEM_MIN_ELEVATION
                )
                gtif = gtif * 2 - 1

        if self._swap:
            return gtif, sat

        return sat, gtif

    def to_img(self, sat_img):
        unnormalized_sat = (sat_img + 1) * 127.5
        unnormalized_sat = unnormalized_sat.permute(1, 2, 0).numpy().astype(np.uint8)
        return unnormalized_sat

    def to_gtif(self, dem):
        return tiff_to_jpg(dem, convert=False)


if __name__ == "__main__":
    import argparse

    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-dem", action="store_true")
    parser.add_argument("--check-sea", action="store_true")
    parser.add_argument("--check-transforms", action="store_true")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--option", choices=["rgb", "dem"])
    args = parser.parse_args()

    if args.sample_dem:
        # Try to grab a dem at random from the dataset and convert it to image
        dataset = AW3D30Dataset(args.dataset, "cpu", normalize=True)
        sat, gtif = dataset[np.random.randint(0, len(dataset))]

        print(gtif.shape, gtif.min(), gtif.max())

        # Convert the gtif to an BW image
        from PIL import Image

        # Find the first image whose amplitude is greater than 100
        tiff_to_jpg(gtif, out_path="gtif.jpg", convert=True)
    elif args.check_sea:
        # Check if the dataset has any sea images
        dataset = AW3D30Dataset(args.dataset, "cpu", normalize=False)
        sea_count = 0

        for sat, gtif in tqdm(dataset, total=len(dataset)):
            np_gtif = gtif.cpu().numpy()

            # Is considered sea if 50% of the depth is below or equal to 0
            is_sea = ((np_gtif <= 0).sum() / np_gtif.size) > 0.5

            if is_sea:
                sea_count += 1

        print(
            f"Sea count: {sea_count}, Total: {len(dataset)}, Percentage: {sea_count / len(dataset) * 100}%"
        )
    elif args.check_transforms:
        dataset = AW3D30Dataset(args.dataset, normalize=True, transforms=True)

        # Sample the same image 10 times
        rand_idx = np.random.randint(0, len(dataset))

        for i in range(10):
            sat, gtif = dataset[rand_idx]

            # Convert the gtif to an BW image
            from PIL import Image

            # Find the first image whose amplitude is greater than 100
            tiff_to_jpg(gtif, out_path=f"gtif_{i}.jpg", convert=True)

            # Convert the sat to an RGB image
            sat_img = dataset.to_img(sat)
            sat_img = Image.fromarray(sat_img)
            sat_img.save(f"sat_{i}.jpg")

    else:
        dataset = AW3D30Dataset(args.dataset, normalize=False)

        # Grab some random indexes from the dataset
        if args.option == "dem":
            indexes = list(range(len(dataset)))
            gtif_tensor = torch.concatenate([dataset[i][1] for i in indexes], dim=0)
            print("GTIF Tensor Shape:", gtif_tensor.shape)
            print("GTIF Tensor Mean:", gtif_tensor.mean())
            print("GTIF Tensor Std:", gtif_tensor.std())
            print("GTIF Tensor Min:", gtif_tensor.min())
            print("GTIF Tensor Max:", gtif_tensor.max())
        elif args.option == "rgb":
            indexes = np.random.randint(0, len(dataset), args.sample_size)
            sat_tensor = torch.cat([dataset[i][0].unsqueeze(0) for i in indexes], dim=0)
            print("SAT Tensor Shape:", sat_tensor.shape)
            print("SAT Tensor Mean:", sat_tensor.mean(dim=(0, 2, 3)))
            print("SAT Tensor Std:", sat_tensor.std(dim=(0, 2, 3)))
