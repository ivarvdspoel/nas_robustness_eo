from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

NUM_CLASSES = 4


def one_hot_mask(mask: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    mask = np.asarray(mask)

    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[0] == num_classes:
            return mask.astype(np.float32)  # already [C,H,W]
        elif mask.shape[-1] == num_classes:
            return mask.transpose(2, 0, 1).astype(np.float32)  # [H,W,C] -> [C,H,W]
        else:
            raise ValueError(f"Expected class-id or one-hot mask, got shape {mask.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    if mask.shape != (128, 128):
        raise ValueError(f"Expected S2 mask shape (128, 128), got {mask.shape}")

    mask = mask.astype(np.int64)
    valid = (mask >= 0) & (mask < num_classes)
    if not np.all(valid):
        raise ValueError(f"Invalid mask values: {np.unique(mask)}")

    onehot = np.eye(num_classes, dtype=np.float32)[mask]  # [H,W,C]
    return onehot.transpose(2, 0, 1)  # [C,H,W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2-root", required=True, help="Folder containing s2_images.npy, s2_masks.npy, manifest.csv")
    parser.add_argument("--out-dir", required=True, help="Output folder")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    s2_root = Path(args.s2_root)
    out_dir = Path(args.out_dir)

    images_out = out_dir / "images_s2_npy"
    masks_out = out_dir / "masks_s2_npy"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    images = np.load(s2_root / "s2_images.npy", mmap_mode="r")
    masks = np.load(s2_root / "s2_masks.npy", mmap_mode="r")
    manifest = pd.read_csv(s2_root / "manifest.csv")

    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)

    if len(images) != len(masks) or len(images) != len(manifest):
        raise ValueError(f"Length mismatch: images={len(images)}, masks={len(masks)}, manifest={len(manifest)}")

    for i, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Writing per-patch S2 npy"):
        patch_id = int(row["patch_index"])

        image_path = images_out / f"image_{patch_id}.npy"
        mask_path = masks_out / f"mask_{patch_id}.npy"

        if not args.overwrite and image_path.exists() and mask_path.exists():
            continue

        img = np.asarray(images[i])
        mask_oh = one_hot_mask(np.asarray(masks[i]))

        if img.shape[-2:] != (128, 128):
            raise ValueError(f"Expected S2 image spatial shape 128x128, got {img.shape}")

        if mask_oh.shape != (4, 128, 128):
            raise ValueError(f"Expected one-hot shape (4,128,128), got {mask_oh.shape}")

        np.save(image_path, img)
        np.save(mask_path, mask_oh)

    manifest.to_csv(out_dir / "manifest_s2_per_patch.csv", index=False)
    print(f"Saved S2 images to: {images_out}")
    print(f"Saved S2 one-hot masks to: {masks_out}")
    print(f"Saved manifest to: {out_dir / 'manifest_s2_per_patch.csv'}")


if __name__ == "__main__":
    main()
