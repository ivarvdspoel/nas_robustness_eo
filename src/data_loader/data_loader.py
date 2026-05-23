# data_loader.py

import math
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

try:
    from lightning import LightningDataModule
except ImportError:
    from pytorch_lightning import LightningDataModule


SQRT_CLIP = math.sqrt(1500.0)

BAND_MEAN = np.array(
    [15.0381, 14.5305, 14.4030, 15.4191, 13.6231, 14.2143, 14.7041, 13.1745],
    dtype=np.float32,
)

BAND_STD = np.array(
    [8.2196, 10.6197, 9.4811, 9.0923, 10.5712, 10.4277, 10.3784, 9.7216],
    dtype=np.float32,
)

NUM_CLASSES = 4


class PhiSatSegDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        selected_band_indices: Optional[Sequence[int]] = None,
        ignore_index: int = 255,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.selected_band_indices = (
            list(selected_band_indices)
            if selected_band_indices is not None
            else list(range(8))
        )
        self.ignore_index = ignore_index

        self.mean = BAND_MEAN[self.selected_band_indices]
        self.std = BAND_STD[self.selected_band_indices]

        self.image_paths = sorted(self.image_dir.glob("image_*.npy"))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No image_*.npy files found in {self.image_dir}")

        self.mask_paths = []

        for image_path in self.image_paths:
            sample_id = image_path.stem.replace("image_", "", 1)
            mask_path = self.mask_dir / f"mask_{sample_id}.npy"

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask for {image_path.name}. Expected: {mask_path}"
                )

            self.mask_paths.append(mask_path)

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"image_paths and mask_paths must have the same length: "
                f"{len(self.image_paths)} != {len(self.mask_paths)}"
            )

    def __len__(self):
        return len(self.image_paths)

    def normalize_(self, x):
        x = x.astype(np.float32)
        x = np.sqrt(np.clip(x, 0.0, None))
        x = np.clip(x, 0.0, SQRT_CLIP)
        x = x[self.selected_band_indices]
        x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)
        return x

    def __getitem__(self, idx):
        x = np.load(self.image_paths[idx])
        y = np.load(self.mask_paths[idx])

        if x.shape != (8, 256, 256):
            raise ValueError(
                f"Expected image shape (8, 256, 256), got {x.shape}: "
                f"{self.image_paths[idx]}"
            )

        if y.shape != (NUM_CLASSES, 256, 256):
            raise ValueError(
                f"Expected preprocessed one-hot mask shape ({NUM_CLASSES}, 256, 256), "
                f"got {y.shape}: {self.mask_paths[idx]}"
            )

        x = self.normalize_(x)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y


class PhiSatSegDataModule(LightningDataModule):
    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        batch_size: int = 8,
        val_split: float = 0.2,
        num_workers: int = 0,
        selected_band_indices: Optional[Sequence[int]] = None,
        ignore_index: int = 255,
        seed: int = 42,
    ):
        super().__init__()

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.test_image_dir = self.image_dir
        self.test_mask_dir = self.mask_dir

        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.selected_band_indices = selected_band_indices
        self.ignore_index = ignore_index
        self.seed = seed

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.input_shape = None
        self.num_classes = None

    def setup(self, stage=None):
        self.dataset = PhiSatSegDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            selected_band_indices=self.selected_band_indices,
            ignore_index=self.ignore_index,
        )

        sample, _ = self.dataset[0]
        self.input_shape = sample.shape
        self.num_classes = NUM_CLASSES

        val_size = int(len(self.dataset) * self.val_split)
        train_size = len(self.dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        if self.test_image_dir is None or self.test_mask_dir is None:
            raise ValueError(
                "test_image_dir and test_mask_dir must be provided to use the test stage."
            )

        self.test_dataset = PhiSatSegDataset(
            image_dir=self.test_image_dir,
            mask_dir=self.test_mask_dir,
            selected_band_indices=self.selected_band_indices,
            ignore_index=self.ignore_index,
        )

        if self.input_shape is None:
            sample, _ = self.test_dataset[0]
            self.input_shape = sample.shape
            self.num_classes = NUM_CLASSES

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None
        )