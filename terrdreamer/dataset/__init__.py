import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm


DEM_MIN_ELEVATION = -283
DEM_MAX_ELEVATION = 7943

class AW3D30Dataset(Dataset):
    def __init__(self,path:Path, device, limit=None):
        self._available_files = list(path.glob("*.npz"))
        
        if limit is not None:
            self._available_files = random.sample(self._available_files, limit)
        
        self._device = device

    def __len__(self):
        return len(self._available_files)
    
    def __getitem__(self, idx):
        data = np.load(self._available_files[idx])

        gtif = torch.tensor(data["GTIF"], dtype=torch.float32)
        sat = torch.tensor(data["SAT"], dtype=torch.float32)

        # Make sure gtif and sat is shaped correctly
        gtif = gtif.unsqueeze(0)
        sat = sat.permute(2, 0, 1)

        # Images also need to be normalized to be between -1 and 1
        sat = (sat /127.5) - 1
        
        # Also normalize the gtif
        gtif = (gtif - DEM_MIN_ELEVATION) / (DEM_MAX_ELEVATION - DEM_MIN_ELEVATION)
        
        # Now we need to make sure that the gtif is between -1 and 1
        gtif = (gtif * 2) - 1
        
        return sat, gtif
    
from PIL import Image

def tiff_to_jpg(tiff_data, convert:bool=False,out_path=None):
    # Unnormalize the gtif
    tiff_data = (tiff_data + 1) / 2
    
    tiff_data = (tiff_data * (DEM_MAX_ELEVATION - DEM_MIN_ELEVATION)) + DEM_MIN_ELEVATION
    
    
    # Normalize the tiff data based on it's min and max, this helps with visualization
    min_value = tiff_data.min()
    max_value = tiff_data.max()
    tiff_data = (tiff_data - min_value) / (max_value - min_value)
    
    img = tiff_data.squeeze().numpy()
    
    # Scale the image
    img = np.maximum(np.minimum(img, 1.0), 0.0)
    
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
    parser.add_argument("--check-sea", action="store_true")
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

        print(f"Sea count: {sea_count}, Total: {len(dataset)}, Percentage: {sea_count / len(dataset) * 100}%")
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
    
