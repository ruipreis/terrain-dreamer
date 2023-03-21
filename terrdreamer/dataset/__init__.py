import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random
from torchvision.transforms import transforms
from PIL import Image

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
    _Satellite_mean = 69.27174377441406
    _Satellite_std = 28.41813850402832

    def __init__(
        self,
        path: Path,
        device,
        normalize=True,
        limit=None,
        for_progan=False,
        new_size=None,
    ):
        self._available_files = list(path.glob("*.npz"))

        if limit is not None:
            self._available_files = random.sample(self._available_files, limit)

        self._device = device
        self._normalize = normalize
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

            gtif = gtif.to(self._device)
            sat = sat.to(self._device)

            if self._normalize:

                # Normalize the data
                sat = (sat - self._Satellite_mean) / self._Satellite_std

                # Apply min max normalization to the dem image
                min_gtif_value = gtif.min()
                max_gtif_value = gtif.max()

                gtif = (gtif - min_gtif_value) / (max_gtif_value - min_gtif_value)

                # Now convert to the range of -1 to 1
                gtif = gtif * 2 - 1

                # Fill nans with -1
                gtif[torch.isnan(gtif)] = -1

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
    else:
        dataset = AW3D30Dataset(args.dataset, "cpu", normalize=False)

        # Compute the mean and std of the dataset for both the satellite and the ground truth
        sat_mean = 0
        sat_std = 0
        gtif_min = np.inf
        gtif_max = -np.inf

        for sat, gtif in tqdm(dataset, total=len(dataset)):
            sat_mean += sat.mean()
            sat_std += sat.std()

            # For the gtif, we want to perform min-max normalization, so we need to compute the min and max
            gtif_min = min(gtif_min, gtif.min())
            gtif_max = max(gtif_max, gtif.max())

        sat_mean /= len(dataset)
        sat_std /= len(dataset)

        print(f"Satellite mean: {sat_mean}")
        print(f"Satellite std: {sat_std}")
        print(f"GTIF min: {gtif_min}")
        print(f"GTIF max: {gtif_max}")
