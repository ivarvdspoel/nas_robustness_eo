import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import random

from dataset_box.perturbation_methods.reobench_perturbations import *

reobench_perturbations = {
    "gaussian_noise": add_gaussian_noise_batch_tensor,
    "salt_pepper": add_salt_pepper_batch_tensor,
    "gaussian_blur": gaussian_blur_batch_tensor,
    "motion_blur": motion_blur_batch_tensor,
    "brightness_contrast": adjust_brightness_contrast_batch_tensor,
    "haze": add_haze_batch_tensor,
}

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='trainval', transform=None, augment_p=0.5, perturbation_type="clean", severity=5):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.augment_p = augment_p
        self.perturbation_type = perturbation_type
        self.severity = severity

        self.classes = ['Background', 'BurntArea', 'Cloud', 'Waterbodies']
        self.num_classes = len(self.classes)

        self.image_paths = []
        self.mask_paths = []
        self.extra_samples = []

        self.images_ram = None
        self.masks_ram = None

        image_dir = os.path.join(self.root_dir, split, 'numpy_images')
        mask_dir = os.path.join(self.root_dir, split, 'numpy_masks')

        image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
        mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

        self.image_paths = [os.path.join(image_dir, f) for f in image_filenames]
        self.mask_paths = [os.path.join(mask_dir, f) for f in mask_filenames]

    def _load_one_file_pair(self, img_path, mask_path):
        image = np.load(img_path)
        if image.shape[-1] == 7:
            image = image.transpose(2, 0, 1)

        mask = np.load(mask_path)
        if mask.shape[-1] == 4:
            mask = mask.transpose(2, 0, 1)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return image, mask

    def add_item(self, x, y):
        x = self._to_tensor(x)
        y = self._to_tensor(y)

        x = self._normalize_image(x)
        y = self._normalize_mask(y)

        self.extra_samples.append((x, y))

    def add_batch(self, x, y):
        x = self._to_tensor(x)
        y = self._to_tensor(y)

        if x.ndim != 4:
            raise ValueError(f"x must be 4D, got shape {tuple(x.shape)}")
        if y.ndim != 4:
            raise ValueError(f"y must be 4D, got shape {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: {x.shape[0]} != {y.shape[0]}")

        if x.shape[-1] == 7 and x.shape[1] != 7:
            x = x.permute(0, 3, 1, 2)
        if y.shape[-1] == 4 and y.shape[1] != 4:
            y = y.permute(0, 3, 1, 2)

        for i in range(x.shape[0]):
            self.extra_samples.append((
                self._normalize_image(x[i]),
                self._normalize_mask(y[i])
            ))

    def _to_tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().clone()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr)
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(arr)}")

    def _normalize_image(self, image):
        if image.ndim != 3:
            raise ValueError(f"Image must be 3D, got shape {tuple(image.shape)}")

        if image.shape[-1] == 7 and image.shape[0] != 7:
            image = image.permute(2, 0, 1)

        return image.float()

    def _normalize_mask(self, mask):
        if mask.ndim != 3:
            raise ValueError(f"Mask must be 3D, got shape {tuple(mask.shape)}")

        if mask.shape[-1] == 4 and mask.shape[0] != 4:
            mask = mask.permute(2, 0, 1)

        return mask.float()

    def __len__(self):
        n_disk_or_ram = len(self.image_paths)
        return n_disk_or_ram + len(self.extra_samples)

    def __getitem__(self, idx):
        n_file_samples = len(self.image_paths)

        if idx < n_file_samples:
            image, mask = self._load_one_file_pair(self.image_paths[idx], self.mask_paths[idx])
        else:
            image, mask = self.extra_samples[idx - n_file_samples]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # only augment training split, with probability p
        chance = torch.rand(1).item()
        if (chance < self.augment_p):
            if self.split == 'trainval' and self.perturbation_type not in ["clean", None]:
                image = self.perturb_image(image)

        return image, mask


    def perturb_image(self, image):
        if self.perturbation_type in ["clean", None]:
            return image

        perturb_fn = reobench_perturbations[self.perturbation_type]
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = perturb_fn(image.unsqueeze(0), severity=self.severity).squeeze(0)
        return image.cpu()


class SegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        batch_size=8,
        num_workers=1,
        transform=None,
        val_split=0.3,
        perturbation_type=None,
        severity=5
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.val_split = val_split
        self.perturbation_type = perturbation_type

        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.severity = severity

        self.setup()

    def prepare_data(self):
        pass

    def perturb_batch(self, x, perturbation="clean"):
        if perturbation == "clean" or perturbation is None:
            return x
        else:
            perturb_fn = reobench_perturbations[perturbation]
            x = perturb_fn(x, severity=self.severity)
            return x
            
    def expand_dataset_with_perturbations(self, dataset, perturbation_type=None, batch_size=None):
        if perturbation_type == "clean" or perturbation_type is None:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = batch_size or self.batch_size
        n_original = len(dataset.image_paths)

        xs, ys = [], []

        for idx in range(n_original):
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)

            is_last = idx == (n_original - 1)
            if len(xs) == batch_size or is_last:
                x_batch = torch.stack(xs, dim=0).to(device)
                y_batch = torch.stack(ys, dim=0).to(device)

                x_perturbed = self.perturb_batch(x_batch, perturbation=perturbation_type)

                # move back to CPU before storing
                dataset.add_batch(x_perturbed.cpu(), y_batch.cpu())

                xs, ys = [], []

    def setup(self, stage='fit'):
        self.full_dataset = SegmentationDataset(
            root_dir=self.root_dir,
            split='trainval',
            transform=self.transform,
            augment_p=0.5,
            perturbation_type=self.perturbation_type,
            severity=self.severity
        )

        self.class_names = self.full_dataset.classes
        self.num_classes = self.full_dataset.num_classes

        sample, _ = self.full_dataset[0]
        self.input_shape = sample.shape

        val_size = int(len(self.full_dataset) * self.val_split)
        train_size = len(self.full_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset,
            [train_size, val_size]
        )

        if stage == 'test':
            self.test_dataset = SegmentationDataset(
                root_dir=self.root_dir,
                split='test',
                transform=self.transform,
                augment_p=0.0,
                severity=self.severity
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None
        )