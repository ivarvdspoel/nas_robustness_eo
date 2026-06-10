# data_loader.py

import math
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


import random

try:
    from lightning import LightningDataModule
except ImportError:
    from pytorch_lightning import LightningDataModule


from src.perturbation_methods.perturbation_methods import *


SQRT_CLIP = math.sqrt(1500.0)

# PhiSat-2 statistics. These are the same values used by the working
# uncommented PhiSat-2 version.
BAND_MEAN = np.array(
    [15.0381, 14.5305, 14.4030, 15.4191, 13.6231, 14.2143, 14.7041, 13.1745],
    dtype=np.float32,
)

BAND_STD = np.array(
    [8.2196, 10.6197, 9.4811, 9.0923, 10.5712, 10.4277, 10.3784, 9.7216],
    dtype=np.float32,
)


# Sentinel-2B statistics.
# Band order: B1, B2, B3, B4, B5, B6, B7
S2B_MEAN = np.array(
    [49.0215, 48.4241, 49.2270, 51.1619, 55.4031, 57.3537, 56.7685],
    dtype=np.float32,
)

S2B_STD = np.array(
    [6.5464, 6.9918, 9.1444, 8.3999, 7.9740, 8.3373, 8.4429],
    dtype=np.float32,
)

S2B_MIN = np.array(
    [0.0, -32701.0, 0.0, 0.0, 0.0, 0.0, -32690.0],
    dtype=np.float32,
)

S2B_MAX = np.array(
    [31463.0, 32724.0, 32218.0, 29106.0, 29044.0, 29031.0, 32424.0],
    dtype=np.float32,
)


def perturb_img(x, perturbation="clean", strength=0):
    if perturbation == "clean":
        return x
    elif perturbation == "noise":
        return perturb_snr(x, snr_factor=strength)
    elif perturbation == "blur":
        return perturb_mtf(x,mtf_nyquist=strength)
    elif perturbation == "haze":
        return perturb_haze(x,t=strength)
    elif perturbation == "brightness":
        return perturb_brightness(x, alpha=strength)
    elif perturbation == "misalignment":
        return perturb_band_misalignment(x, max_shift_px=strength)
    else:
        raise ValueError(
            f"Unknown perturbation {perturbation!r}. "
            "Expected one of: 'noise', 'blur', 'misalignment', "
            "'haze', 'brightness', 'none'."
        )


NUM_CLASSES = 4


class PhiSatSegDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        selected_band_indices: Optional[Sequence[int]] = None,
        ignore_index: int = 255,
        perturbation="clean",
        strength = 0,
        to_normalize=True,
        stage=None
    ):
        self.perturbation = perturbation
        self.to_normalize = to_normalize
        self.strength = strength

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.stage = stage

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

    def normalize_batch(self, x):
        """
        x: torch.Tensor [B, C, H, W], raw unnormalized batch
        returns normalized torch.Tensor [B, selected_C, H, W]
        """
        x = x.float()

        x = torch.sqrt(torch.clamp(x, min=0.0))
        x = torch.clamp(x, min=0.0, max=SQRT_CLIP)

        x = x[:, self.selected_band_indices, :, :]

        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)[None, :, None, None]
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)[None, :, None, None]

        x = (x - mean) / (std + 1e-6)
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

        if self.perturbation != "clean" and self.stage == "train":
            if random.random() < 0.5:
                x = perturb_img(x, self.perturbation, self.strength)
                
        if self.to_normalize:
            x = self.normalize_(x)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y


class Sentinel2SegDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        selected_band_indices: Optional[Sequence[int]] = None,
        ignore_index: int = 255,
        perturbation="clean",
        strength=0,
        to_normalize=True,
        stage=None,
    ):
        self.perturbation = perturbation
        self.strength = strength
        self.to_normalize = to_normalize
        self.stage = stage

        # Sentinel-2B band statistics
        # Band order: B1, B2, B3, B4, B5, B6, B7
        self.s2b_mean = S2B_MEAN
        self.s2b_std = S2B_STD
        self.s2b_min = S2B_MIN
        self.s2b_max = S2B_MAX

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.selected_band_indices = (
            list(selected_band_indices)
            if selected_band_indices is not None
            else list(range(7))
        )
        self.ignore_index = ignore_index

        self.mean = S2B_MEAN[self.selected_band_indices]
        self.std = S2B_STD[self.selected_band_indices]
        self.min = S2B_MIN[self.selected_band_indices]
        self.max = S2B_MAX[self.selected_band_indices]

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
        x = x[self.selected_band_indices]

        # Clip to observed raw min/max per selected band.
        x = np.clip(
            x,
            self.min[:, None, None],
            self.max[:, None, None],
        )

        # Z-score normalization per selected band.
        x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)

        return x.astype(np.float32)

    def normalize_batch(self, x):
        """
        x: torch.Tensor [B, C, H, W], raw unnormalized batch
        returns normalized torch.Tensor [B, selected_C, H, W]
        """
        x = x.float()

        x = x[:, self.selected_band_indices, :, :]

        min_v = torch.as_tensor(self.min, device=x.device, dtype=x.dtype)[None, :, None, None]
        max_v = torch.as_tensor(self.max, device=x.device, dtype=x.dtype)[None, :, None, None]
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)[None, :, None, None]
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)[None, :, None, None]

        x = torch.clamp(x, min=min_v, max=max_v)
        x = (x - mean) / (std + 1e-6)

        return x



    def __getitem__(self, idx):
        x = np.load(self.image_paths[idx])
        y = np.load(self.mask_paths[idx])

        if x.shape != (7, 128, 128):
            raise ValueError(
                f"Expected image shape (7, 128, 128), got {x.shape}: "
                f"{self.image_paths[idx]}"
            )

        if y.shape != (NUM_CLASSES, 128, 128):
            raise ValueError(
                f"Expected preprocessed one-hot mask shape ({NUM_CLASSES}, 128, 128), "
                f"got {y.shape}: {self.mask_paths[idx]}"
            )

        if self.perturbation != "clean" and self.stage == "train":
            if random.random() < 0.5:
                x = perturb_img(x, self.perturbation, self.strength)

        if self.to_normalize:
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
        perturbation="clean",
        strength = 0
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

        self.perturbation=perturbation
        self.strength = strength

    def setup(self, stage=None):
        train_full = PhiSatSegDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            selected_band_indices=self.selected_band_indices,
            ignore_index=self.ignore_index,
            perturbation=self.perturbation,
            strength=self.strength,
            to_normalize=True,
            stage="train",
        )

        val_full = PhiSatSegDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            selected_band_indices=self.selected_band_indices,
            ignore_index=self.ignore_index,
            perturbation="clean",
            strength=0,
            to_normalize=True,
            stage="val",
        )

        sample, _ = train_full[0]
        self.input_shape = sample.shape
        self.num_classes = NUM_CLASSES

        val_size = int(len(train_full) * self.val_split)
        train_size = len(train_full) - val_size

        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(train_full), generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        self.train_dataset = Subset(train_full, train_indices)
        self.val_dataset = Subset(val_full, val_indices)
        self.dataset = train_full

        if self.test_image_dir is None or self.test_mask_dir is None:
            raise ValueError(
                "test_image_dir and test_mask_dir must be provided to use the test stage."
            )
            
        self.test_dataset = PhiSatSegDataset(
            image_dir=self.test_image_dir,
            mask_dir=self.test_mask_dir,
            selected_band_indices=self.selected_band_indices,
            ignore_index=self.ignore_index,
            perturbation=self.perturbation,
            strength=self.strength,
            to_normalize=False,
            stage="test",
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
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


class Sentinel2SegDataModule(LightningDataModule):
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
        perturbation = "clean",
        strength = 0
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
        self.test_dataset = None
        self.input_shape = None
        self.num_classes = None
        self.perturbation = perturbation
        self.strength = strength

    def setup(self, stage=None):
        train_full = Sentinel2SegDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            selected_band_indices=self.selected_band_indices,
            ignore_index=self.ignore_index,
            perturbation=self.perturbation,
            strength=self.strength,
            to_normalize=True,
            stage="train",
        )

        val_full = Sentinel2SegDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            selected_band_indices=self.selected_band_indices,
            ignore_index=self.ignore_index,
            perturbation="clean",
            strength=0,
            to_normalize=True,
            stage="val",
        )

        sample, _ = train_full[0]
        self.input_shape = sample.shape
        self.num_classes = NUM_CLASSES

        val_size = int(len(train_full) * self.val_split)
        train_size = len(train_full) - val_size

        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(train_full), generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        self.train_dataset = Subset(train_full, train_indices)
        self.val_dataset = Subset(val_full, val_indices)
        self.dataset = train_full

        if self.test_image_dir is None or self.test_mask_dir is None:
            raise ValueError(
                "test_image_dir and test_mask_dir must be provided to use the test stage."
            )


        self.test_dataset = Sentinel2SegDataset(
            image_dir=self.test_image_dir,
            mask_dir=self.test_mask_dir,
            selected_band_indices=self.selected_band_indices,
            ignore_index=self.ignore_index,
            perturbation=self.perturbation,
            strength=self.strength,
            to_normalize=False,
            stage="test",
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
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
