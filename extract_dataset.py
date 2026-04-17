from pathlib import Path
import numpy as np
import zarr
from tqdm import tqdm


#ZARR_ROOT = Path("/shared/home/ivanderspoel/dataset/OEOBench-Burnt_Area_Dataset/burned.zarr.zip")
ZARR_ROOT = Path("/tmp/ivar_data/burned.zarr.zip")
OUT_ROOT = Path("/tmp/ivar_data/data")  # where the npy dataset will go


# Open Zarr root
root = zarr.open_group(ZARR_ROOT, mode="r")

# Process each split
for split in ["trainval", "test"]:
    split_ids = sorted(root[split].keys())
    
    img_out_dir = OUT_ROOT / split / "numpy_images"
    mask_out_dir = OUT_ROOT / split / "numpy_masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting split: {split} ({len(split_ids)} samples)")
    
    for sid in tqdm(split_ids):
        sample = root[split][sid]
        
        # Load Zarr arrays into numpy
        img = sample["img"][:]
        label = sample["label"][:]
        
        # Save as .npy
        np.save(img_out_dir / f"{sid}.npy", img)
        np.save(mask_out_dir / f"{sid}.npy", label)

print("Conversion complete!")
print(f"Dataset saved in: {OUT_ROOT}")