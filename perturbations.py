import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import timm
import numpy as np
import tempfile
import subprocess
import os

import sys
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))  

# Now import from pynas
from pynas.scripts.dataloader import SegmentationDataModule



# exit(1)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL =================
ckpt_path = Path("/home/s3167445/msc_thesis_ivar/pynas/notebooks/checkpoints/resnet18-seg-epoch=00-val_miou=0.9625.ckpt")

class ResNet18Segmentation(pl.LightningModule):
    def __init__(self, num_classes=4, in_channels=7):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18", pretrained=True, features_only=True, in_chans=in_channels
        )
        in_chs = self.backbone.feature_info[-1]["num_chs"]
        self.seg_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_chs, num_classes, kernel_size=1),
            torch.nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]
        return self.seg_head(feats)

# ================= LOAD CHECKPOINT =================
root_dir = '/local/s3167445/data'
dm = SegmentationDataModule(root_dir, batch_size=2, num_workers=4, transform=None)
dm.setup(stage='test')  # load test dataset

# Detect input channels dynamically
sample_img, _ = dm.test_dataset[0]
in_ch = sample_img.shape[0]  # CHW format
print("Detected input channels:", in_ch)

model = ResNet18Segmentation(num_classes=4, in_channels=in_ch)
state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ================= NOISE FUNCTION =================
EXEC_PATH = "/home/s3167445/msc_thesis_ivar/phisat2_unix.bin"


def apply_noise_to_npy(input_arr: np.ndarray, calculation="SNR"):
    """
    Apply Phisat SNR/PSF noise to a single .npy file.
    Pads 7-channel input to 8 channels for Phisat.
    Returns noisy 7-channel CHW torch tensor.
    """

    # Ensure HWC
    if input_arr.ndim == 3 and input_arr.shape[0] == 7:  # CHW -> HWC
        input_arr = input_arr.transpose(1,2,0)

    # Pad dummy channel if needed
    if input_arr.shape[2] == 7:
        input_arr = np.concatenate([input_arr, np.zeros((*input_arr.shape[:2],1), dtype=input_arr.dtype)], axis=2)

    # Save temp
    with tempfile.TemporaryDirectory(prefix="phisat2_calc", suffix=calculation) as tmpdir:
        tmp_input = os.path.join(tmpdir, "input.npy")
        tmp_output = os.path.join(tmpdir, "output.npy")
        np.save(tmp_input, input_arr)

        subprocess.run([EXEC_PATH, calculation, tmp_input, tmp_output], check=True)

        noisy_arr = np.load(tmp_output)

    # Phisat may return (H,W,C) or (1,H,W,C). Flatten batch dimension if needed
    if noisy_arr.ndim == 4 and noisy_arr.shape[0] == 1:
        noisy_arr = noisy_arr[0]

    # Remove dummy channel if present
    if noisy_arr.shape[2] == 8:
        noisy_arr = noisy_arr[..., :7]

    # Convert to CHW for PyTorch
    return torch.tensor(noisy_arr.transpose(2,0,1), dtype=torch.float32)


# ================= mIoU FUNCTION =================
def compute_miou(model, dataloader, num_classes=4, device=device):
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            # Convert one-hot to class indices if needed
            if y.ndim == 4:
                y = torch.argmax(y, dim=1)
            y = y.long().to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            for cls in range(num_classes):
                inter = ((preds == cls) & (y == cls)).sum()
                uni = ((preds == cls) | (y == cls)).sum()
                intersection[cls] += inter
                union[cls] += uni

    return (intersection / union).mean().item()

# ================= CLEAN DATA =================
clean_loader = DataLoader(dm.test_dataset, batch_size=2, shuffle=False, num_workers=4)
miou_clean = compute_miou(model, clean_loader)
print(f"mIoU on clean test set: {miou_clean:.4f}")

# ================= NOISY DATA =================
class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, calculation='SNR'):
        self.base_dataset = base_dataset
        self.calculation = calculation

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, mask = self.base_dataset[idx]
        noisy_img = apply_noise_to_npy(img, calculation=self.calculation)
        return torch.tensor(noisy_img, dtype=torch.float32), mask

noisy_dataset = NoisyDataset(dm.test_dataset, calculation='SNR')
noisy_loader = DataLoader(noisy_dataset, batch_size=2, shuffle=False, num_workers=4)
miou_noisy = compute_miou(model, noisy_loader)
print(f"mIoU on noisy test set: {miou_noisy:.4f}")
