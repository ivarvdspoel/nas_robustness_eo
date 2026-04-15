import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='TrainVal', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            split (string): One of ['TrainVal', 'Test'] to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.classes = ['Background', 'BurntArea', 'Cloud', 'Waterbodies']
        self.num_classes = len(self.classes)

        self.image_paths = []
        self.mask_paths = []

        # in-memory samples added during runtime
        self.extra_samples = []

        image_dir = os.path.join(self.root_dir, split, 'numpy_images')
        mask_dir = os.path.join(self.root_dir, split, 'numpy_masks')

        image_filenames = os.listdir(image_dir)
        mask_filenames = os.listdir(mask_dir)

        for image_filename in image_filenames:
            if image_filename.endswith('.npy'):
                self.image_paths.append(os.path.join(image_dir, image_filename))

        for mask_filename in mask_filenames:
            if mask_filename.endswith('.npy'):
                self.mask_paths.append(os.path.join(mask_dir, mask_filename))

        self.image_paths.sort()
        self.mask_paths.sort()

    def add_item(self, x, y):
        """
        Add one in-memory sample.

        x: torch.Tensor or np.ndarray, shape [C,H,W] or [H,W,C]
        y: torch.Tensor or np.ndarray, shape [C,H,W] or [H,W,C]
        """
        x = self._to_tensor(x)
        y = self._to_tensor(y)

        x = self._normalize_image(x)
        y = self._normalize_mask(y)

        self.extra_samples.append((x, y))

    def add_batch(self, x, y):
        """
        Add a batch of in-memory samples.

        x: torch.Tensor or np.ndarray, shape [B,C,H,W] or [B,H,W,C]
        y: torch.Tensor or np.ndarray, shape [B,C,H,W] or [B,H,W,C]
        """
        x = self._to_tensor(x)
        y = self._to_tensor(y)

        if x.ndim != 4:
            raise ValueError(f"x must be 4D, got shape {tuple(x.shape)}")
        if y.ndim != 4:
            raise ValueError(f"y must be 4D, got shape {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: {x.shape[0]} != {y.shape[0]}")

        # convert [B,H,W,C] -> [B,C,H,W] when needed
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

        # [H,W,C] -> [C,H,W]
        if image.shape[-1] == 7 and image.shape[0] != 7:
            image = image.permute(2, 0, 1)

        return image.float()

    def _normalize_mask(self, mask):
        if mask.ndim != 3:
            raise ValueError(f"Mask must be 3D, got shape {tuple(mask.shape)}")

        # [H,W,C] -> [C,H,W]
        if mask.shape[-1] == 4 and mask.shape[0] != 4:
            mask = mask.permute(2, 0, 1)

        return mask.float()

    def __len__(self):
        return len(self.image_paths) + len(self.extra_samples)

    def __getitem__(self, idx):
        n_file_samples = len(self.image_paths)

        if idx < n_file_samples:
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]

            image = np.load(img_path)
            if image.shape[-1] == 7:
                image = image.transpose(2, 0, 1)

            mask = np.load(mask_path)
            if mask.shape[-1] == 4:
                mask = mask.transpose(2, 0, 1)

            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).float()
        else:
            image, mask = self.extra_samples[idx - n_file_samples]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class SegmentationDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size=8, num_workers=1, transform=None, val_split=0.3):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.val_split = val_split

        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage='fit'):
        self.full_dataset = SegmentationDataset(
            root_dir=self.root_dir,
            split='TrainVal',
            transform=self.transform,
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
                split='Test',
                transform=self.transform,
            )

    def _resplit_train_val(self):
        val_size = int(len(self.full_dataset) * self.val_split)
        train_size = len(self.full_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset,
            [train_size, val_size]
        )

    def add_item(self, x, y):
        """
        Add one in-memory sample to the dataset, then rebuild the train/val split.
        """
        self.full_dataset.add_item(x, y)
        self._resplit_train_val()

    def add_batch(self, x, y):
        """
        Add a batch of in-memory samples to the dataset, then rebuild the train/val split.
        """
        self.full_dataset.add_batch(x, y)
        self._resplit_train_val()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )