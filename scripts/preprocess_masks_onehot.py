# scripts/preprocess_masks_onehot.py

from pathlib import Path
import argparse

import numpy as np
from tqdm import tqdm


NUM_CLASSES = 4


def one_hot_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)

    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            raise ValueError(f"Expected class-id mask, got shape {mask.shape}")

    if mask.shape != (256, 256):
        raise ValueError(f"Expected mask shape (256, 256), got {mask.shape}")

    mask = mask.astype(np.int64)

    valid = (mask >= 0) & (mask < NUM_CLASSES)
    if not np.all(valid):
        raise ValueError(f"Invalid mask values: {np.unique(mask)}")

    # [H, W] -> [H, W, C] -> [C, H, W]
    onehot = np.eye(NUM_CLASSES, dtype=np.float32)[mask]
    onehot = onehot.transpose(2, 0, 1)

    return onehot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(mask_dir.glob("mask_*.npy"))

    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No mask_*.npy files found in {mask_dir}")

    for mask_path in tqdm(mask_paths, desc="Preprocessing masks"):
        out_path = out_dir / mask_path.name

        if out_path.exists() and not args.overwrite:
            continue

        mask = np.load(mask_path)
        mask_onehot = one_hot_mask(mask)

        if mask_onehot.shape != (4, 256, 256):
            raise ValueError(
                f"Expected one-hot shape (4, 256, 256), got {mask_onehot.shape}"
            )

        np.save(out_path, mask_onehot)

    print(f"Saved one-hot masks to: {out_dir}")


if __name__ == "__main__":
    main()