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
#device = torch.device("cpu")

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
# EXEC_PATH = "/home/s3167445/msc_thesis_ivar/phisat2_unix.bin"


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

def compute_consistency(model, dataloader, reference_preds, device=device):
    """
    reference_preds: List of tensors containing clean predictions for each batch.
    """
    total_matching_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.float().to(device)
            
            # Get predictions for noisy images
            logits = model(x)
            preds_noisy = torch.argmax(logits, dim=1)
            
            # Get the pre-computed clean predictions for this batch
            preds_clean = reference_preds[batch_idx].to(device)
            
            # Calculate matches
            matching = (preds_noisy == preds_clean).sum().item()
            total_matching_pixels += matching
            total_pixels += preds_clean.numel()

    return total_matching_pixels / total_pixels

print("Pre-computing reference predictions on clean data...")
clean_loader = DataLoader(dm.test_dataset, batch_size=2, shuffle=False, num_workers=4)
reference_preds = []

model.eval()
with torch.no_grad():
    for x, _ in clean_loader:
        x = x.float().to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu() # Move to CPU to save GPU memory
        reference_preds.append(preds)

strengths = np.arange(0.1, 1.1, 0.1)
consistency_results = []

print(f"\n{'Strength':<10} | {'Consistency':<12}")
print("-" * 30)


import matplotlib.pyplot as plt

def apply_snr_noise_native(img, snr_target=174, l_ref=100, strength=1.0):
    if strength == 0:
        return img
    
    # Check if it's a torch tensor or numpy array
    if torch.is_tensor(img):
        img_np = img.detach().cpu().numpy()
    else:
        img_np = img

    # Generate noise on CPU (saves VRAM)
    noise = np.random.normal(size=img_np.shape).astype(np.float32)
    
    # PhiSat formula: Noise = (L_ref / SNR) * N(0,1)
    noise_component = (l_ref / snr_target) * noise
    noisy_img = img_np + (strength * noise_component)
    
    return torch.from_numpy(noisy_img)

class DynamicNoisyDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, strength):
        self.base_dataset = base_dataset
        self.strength = strength
    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, idx):
        img, mask = self.base_dataset[idx]
        # SNR values vary by band, using a mean 174 as example
        noisy_img = apply_snr_noise_native(img, snr_target=174, strength=self.strength)
        return noisy_img, mask

for s in strengths:
    temp_ds = DynamicNoisyDataset(dm.test_dataset, strength=s)
    # Ensure shuffle=False so batches align with reference_preds
    temp_loader = DataLoader(temp_ds, batch_size=2, shuffle=False, num_workers=4)
    
    current_consistency = compute_consistency(model, temp_loader, reference_preds)
    consistency_results.append(current_consistency)
    print(f"{s:.1f}        | {current_consistency:.4%}")

# ================= 3. PLOT RESULTS =================
plt.figure(figsize=(8, 5))
plt.plot(strengths, consistency_results, marker='s', linestyle='--', color='green')
plt.title("Prediction Consistency: Clean vs. Noisy")
plt.xlabel("Perturbation Strength")
plt.ylabel("Consistency (%)")
plt.ylim(0, 1.05)
plt.grid(True)
plt.savefig("consistency_curve.png")

exit(1)

# ================= CLEAN DATA =================
# clean_loader = DataLoader(dm.test_dataset, batch_size=2, shuffle=False, num_workers=4)
# miou_clean = compute_miou(model, clean_loader)
# print(f"mIoU on clean test set: {miou_clean:.4f}")

# # ================= NOISY DATA =================
# class NoisyDataset(torch.utils.data.Dataset):
#     def __init__(self, base_dataset, calculation='SNR'):
#         self.base_dataset = base_dataset
#         self.calculation = calculation

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         img, mask = self.base_dataset[idx]
#         noisy_img = apply_noise_to_npy(img, calculation=self.calculation)
#         return torch.tensor(noisy_img, dtype=torch.float32), mask

# noisy_dataset = NoisyDataset(dm.test_dataset, calculation='SNR')
# noisy_loader = DataLoader(noisy_dataset, batch_size=2, shuffle=False, num_workers=4)
# miou_noisy = compute_miou(model, noisy_loader)
# print(f"mIoU on noisy test set: {miou_noisy:.4f}")


# ================= PERTURBATION LOOP =================
strengths = np.arange(0.1, 1.1, 0.1)
miou_results = []

print(f"{'Strength':<10} | {'mIoU':<10}")
print("-" * 25)

for s in strengths:
    # We wrap the dataset with our strength-specific noise
    class DynamicNoisyDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, strength):
            self.base_dataset = base_dataset
            self.strength = strength
        def __len__(self): return len(self.base_dataset)
        def __getitem__(self, idx):
            img, mask = self.base_dataset[idx]
            # SNR values vary by band, using a mean 174 as example
            noisy_img = apply_snr_noise_native(img, snr_target=174, strength=self.strength)
            return noisy_img, mask

    temp_ds = DynamicNoisyDataset(dm.test_dataset, strength=s)
    temp_loader = DataLoader(temp_ds, batch_size=2, num_workers=4)
    
    current_miou = compute_miou(model, temp_loader)
    miou_results.append(current_miou)
    print(f"{s:.1f}        | {current_miou:.4f}")

# ================= PLOT RESULTS =================
print(miou_results)
plt.figure(figsize=(8, 5))
plt.plot(strengths, miou_results, marker='o', linestyle='-', color='b')
plt.title("Model Robustness: mIoU vs. SNR Degradation")
plt.xlabel("Perturbation Strength (0=BOL, 1=EOL)")
plt.ylabel("mIoU")
plt.grid(True)
plt.savefig("robustness_curve.png")
#plt.show()
