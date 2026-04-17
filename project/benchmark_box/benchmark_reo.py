import sys
import os
import csv
import importlib
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(".."))

from dataset_box.data_loader import *
from model_box.baselines.resnet_baseline.baseline_cnn import *
from model_box.baselines.vit_baseline.vit_seg import *
import evaluation_box.evaluation_functions
importlib.reload(evaluation_box.evaluation_functions)
from evaluation_box.evaluation_functions import *
from dataset_box.perturbation_methods.reobench_perturbations import *

# =========================
# Setup
# =========================
root_dir = '/local/s3167445/data'
batch_size = 2
num_workers = 4
num_classes = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dm = SegmentationDataModule(root_dir, batch_size=batch_size, num_workers=num_workers, transform=None)
dm.setup(stage='test')

sample_img, _ = dm.test_dataset[0]
in_ch = sample_img.shape[0]
print("Detected input channels:", in_ch)

test_loader = DataLoader(dm.test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# =========================
# Load models
# =========================
models = {}

# ckpt model


# PyNAS
pynas = torch.jit.load('model_box/saved_models/PNAS_NVIDIA_jetson_AGX_orin.pt', map_location=device)
pynas.to(device)
pynas.eval()
models["PyNAS"] = pynas

# ResNet
resnet = ResNet18UNet(in_channels=in_ch, num_classes=num_classes)
resnet.load_state_dict(torch.load("model_box/saved_models/resnet.pt", map_location=device))
resnet.to(device)
resnet.eval()
models["ResNet"] = resnet

# ViT
vit = ViTSegmentation(in_chans=in_ch, num_classes=num_classes)
vit.load_state_dict(torch.load("model_box/saved_models/vit.pt", map_location=device))
vit.to(device)
vit.eval()
models["ViT"] = vit


# =========================
# Perturbation registry
# =========================
reobench_perturbations = {
    # "gaussian_noise": add_gaussian_noise_batch_tensor,
    # "salt_pepper": add_salt_pepper_batch_tensor,
    # "gaussian_blur": gaussian_blur_batch_tensor,
    "motion_blur": motion_blur_batch_tensor,
    "brightness_contrast": adjust_brightness_contrast_batch_tensor,
    "haze": add_haze_batch_tensor,
}

severities = [5]


# =========================
# Helpers
# =========================
def prepare_labels(y: torch.Tensor) -> torch.Tensor:
    """
    Converts labels to class-index format [B,H,W] if they are one-hot [B,C,H,W].
    """
    if y.ndim == 4:
        y = torch.argmax(y, dim=1)
    return y.long()


def get_predictions(model, dataloader, device="cpu", perturb_fn=None, severity=None):
    """
    Runs inference, optionally applying a perturbation to each batch before prediction.

    Returns:
        preds:  tensor [N,H,W] or concatenated batch predictions
        labels: tensor [N,H,W]
    """
    preds_all = []
    labels_all = []

    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            y = prepare_labels(y).to(device)

            if perturb_fn is not None:
                x = perturb_fn(x, severity=severity)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            preds_all.append(preds.cpu())
            labels_all.append(y.cpu())

    if len(labels_all) == 0:
        raise ValueError("Dataloader produced no batches.")

    preds = torch.cat(preds_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    return preds, labels


def safe_compute_miou(preds, labels):
    """
    Wrapper in case your compute_miou_ signature or behavior changes.
    """
    return compute_miou_(preds=preds, labels=labels)


def safe_compute_consistency(clean_preds, perturbed_preds):
    """
    Consistency = fraction of pixels unchanged between clean and perturbed predictions.
    Replace this with your own function if you already have one in evaluation_functions.
    """
    if clean_preds.shape != perturbed_preds.shape:
        raise ValueError(f"Shape mismatch: {clean_preds.shape} vs {perturbed_preds.shape}")
    return (clean_preds == perturbed_preds).float().mean().item()


# =========================
# Clean baseline predictions
# =========================
print("\nComputing clean predictions...")
clean_preds = {}
labels_ref = None

for model_name, model in models.items():
    preds, labels = get_predictions(model, test_loader, device=device, perturb_fn=None, severity=None)
    clean_preds[model_name] = preds

    if labels_ref is None:
        labels_ref = labels
    else:
        if not torch.equal(labels_ref, labels):
            raise RuntimeError(f"Label mismatch detected for model {model_name}.")

    miou_clean = safe_compute_miou(preds, labels_ref)
    print(f"{model_name} clean mIoU: {miou_clean}")


# =========================
# Benchmark loop
# =========================
results = []

print("\nRunning REO perturbation benchmark...")
for perturb_name, perturb_fn in reobench_perturbations.items():
    print(f"\n=== Perturbation: {perturb_name} ===")

    for severity in severities:
        print(f"  Severity {severity}")

        for model_name, model in models.items():
            preds_pert, labels = get_predictions(
                model=model,
                dataloader=test_loader,
                device=device,
                perturb_fn=perturb_fn,
                severity=severity
            )

            miou = safe_compute_miou(preds_pert, labels)
            consistency = safe_compute_consistency(clean_preds[model_name], preds_pert)

            row = {
                "perturbation": perturb_name,
                "severity": severity,
                "model": model_name,
                "mIoU": float(miou),
                "consistency": float(consistency),
            }
            results.append(row)

            print(
                f"    {model_name:<8} | "
                f"mIoU: {miou:.4f} | "
                f"consistency: {consistency:.4f}"
            )


# =========================
# Save results
# =========================
output_csv = "reobench_model_eval.csv"
with open(output_csv, mode="w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["perturbation", "severity", "model", "mIoU", "consistency"]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to: {output_csv}")


# =========================
# Optional summary print
# =========================
print("\nSummary by perturbation/model:")
summary = defaultdict(list)
for row in results:
    key = (row["perturbation"], row["model"])
    summary[key].append(row["mIoU"])

for (perturbation, model_name), vals in summary.items():
    avg_miou = sum(vals) / len(vals)
    print(f"{perturbation:<24} {model_name:<8} avg mIoU: {avg_miou:.4f}")