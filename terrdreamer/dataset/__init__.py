import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random

class AW3D30Dataset(Dataset):
    _Satellite_mean= 69.27174377441406
    _Satellite_std= 28.41813850402832
    _GTIF_min= -9999.0
    _GTIF_max= 7943.0

    def __init__(self,path:Path, device, normalize=True, limit=None):
        self._available_files = list(path.glob("*.npz"))
        
        if limit is not None:
            self._available_files = random.sample(self._available_files, limit)
        
        self._device = device
        self._normalize = normalize

    def __len__(self):
        return len(self._available_files)
    
    def __getitem__(self, idx):
        data = np.load(self._available_files[idx])

        gtif = torch.tensor(data["GTIF"], dtype=torch.float32)
        sat = torch.tensor(data["SAT"], dtype=torch.float32)

        # Make sure gtif and sat is shaped correctly
        gtif = gtif.unsqueeze(0)
        sat = sat.permute(2, 0, 1)

        gtif  = gtif.to(self._device)
        sat = sat.to(self._device)

        if self._normalize:
            # Normalize the data
            sat = (sat - self._Satellite_mean) / self._Satellite_std
            
            # The gtif file uses min-max normalization, the range is now from 0 to 1
            gtif = (gtif - self._GTIF_min) / (self._GTIF_max - self._GTIF_min)

        return sat, gtif
    
from PIL import Image

def tiff_to_jpg(tiff_data,min_value, max_value,scale:int=5,convert:bool=False,out_path=None):
    # Unnormalize the gtif
    img = (tiff_data * (max_value - min_value) + min_value).squeeze().numpy()
    img = np.where(img < 0, 0, img)
    img /= max_value
    
    # Scale the image
    img = np.minimum(img * scale, 1.0)
    
    img = (img * 255).astype(np.uint8)
    
    if convert:
        img = Image.fromarray(img)
        img = img.convert("L")
        img.save(out_path)

    return img


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
        for i in range(len(dataset)):
            _, gtif = dataset[i]
            img = tiff_to_jpg(gtif, AW3D30Dataset._GTIF_min, AW3D30Dataset._GTIF_max)
            
            min_value = img.min()
            max_value = img.max()
            print(i, min_value, max_value)
            
            if (max_value - min_value) > 25 and min_value >= 20:
                break
            
        # Convert the gtif to an image
        img = Image.fromarray(img)
        img = img.convert("L")
        img.save("gtif.jpg")
        
        
        
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
    
