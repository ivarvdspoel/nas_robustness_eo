#!/usr/bin/env python3
"""
Train a ResNet-18 + U-Net decoder semantic-segmentation baseline on either
PhiSat-2 or Sentinel-2 patches.

Expected dataloader:
    from src.data_loader.data_loader import PhiSatSegDataModule, Sentinel2SegDataModule

Default dataset paths follow the NAS main.py setup:
    PhiSat-2:   8 x 256 x 256 images, 4 x 256 x 256 one-hot masks
    Sentinel-2: 7 x 128 x 128 images, 4 x 128 x 128 one-hot masks

Model:
    timm ResNet-18 encoder, pretrained=False, features_only=True
    lightweight U-Net-style decoder with skip connections

Training:
    Adam, lr=1e-3, focal loss, 10 epochs by default
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from src.data_loader.data_loader import PhiSatSegDataModule, Sentinel2SegDataModule


NUM_CLASSES = 4
IGNORE_INDEX = 255


# -----------------------------------------------------------------------------
# Loss and metrics
# -----------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        valid = target != self.ignore_index
        ce = ce[valid]
        if ce.numel() == 0:
            return logits.sum() * 0.0

        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


@torch.no_grad()
def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    valid_class = (target >= 0) & (target < num_classes)
    pred = pred[valid_class]
    target = target[valid_class]

    idx = target * num_classes + pred
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


@torch.no_grad()
def metrics_from_cm(cm: torch.Tensor) -> dict[str, torch.Tensor]:
    cm = cm.float()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    union = tp + fp + fn
    valid = union > 0
    iou = torch.where(valid, tp / union.clamp_min(1.0), torch.nan)

    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)

    pixel_acc = tp.sum() / cm.sum().clamp_min(1.0)

    return {
        "miou": torch.nanmean(iou),
        "macro_f1": torch.nanmean(f1),
        "pixel_acc": pixel_acc,
    }


# -----------------------------------------------------------------------------
# ResNet-18 U-Net model
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet18UNet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        num_classes: int = NUM_CLASSES,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 32),
    ):
        super().__init__()

        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        encoder_channels = self.encoder.feature_info.channels()

        if len(decoder_channels) != len(encoder_channels):
            raise ValueError(
                "decoder_channels must have the same length as the timm feature pyramid. "
                f"Got {len(decoder_channels)} decoder channels and {len(encoder_channels)} encoder features."
            )

        self.bottleneck = ConvBlock(encoder_channels[-1], decoder_channels[0])

        up_blocks = []
        in_c = decoder_channels[0]
        for skip_c, out_c in zip(reversed(encoder_channels[:-1]), decoder_channels[1:]):
            up_blocks.append(UpBlock(in_channels=in_c, skip_channels=skip_c, out_channels=out_c))
            in_c = out_c

        self.up_blocks = nn.ModuleList(up_blocks)
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feats = self.encoder(x)

        y = self.bottleneck(feats[-1])
        for up, skip in zip(self.up_blocks, reversed(feats[:-1])):
            y = up(y, skip)

        logits = self.seg_head(y)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


# -----------------------------------------------------------------------------
# Lightning module
# -----------------------------------------------------------------------------

class ResNet18UNetLit(pl.LightningModule):
    def __init__(
        self,
        in_chans: int,
        num_classes: int = NUM_CLASSES,
        lr: float = 1e-3,
        focal_gamma: float = 2.0,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResNet18UNet(in_chans=in_chans, num_classes=num_classes)
        self.loss_fn = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)

        self.register_buffer(
            "val_cm",
            torch.zeros(num_classes, num_classes, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _target_to_index(self, y: torch.Tensor) -> torch.Tensor:
        """
        Supports both:
          - one-hot masks: [B, C, H, W]
          - class-index masks: [B, H, W] or [B, 1, H, W]
        """
        if y.ndim == 4 and y.shape[1] == self.hparams.num_classes:
            valid = y.sum(dim=1) > 0
            target = y.argmax(dim=1).long()
            target = torch.where(
                valid,
                target,
                torch.full_like(target, int(self.hparams.ignore_index)),
            )
            return target

        if y.ndim == 4 and y.shape[1] == 1:
            return y[:, 0].long()

        if y.ndim == 3:
            return y.long()

        raise ValueError(f"Unsupported target shape: {tuple(y.shape)}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        target = self._target_to_index(y)
        logits = self(x)
        loss = self.loss_fn(logits, target)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=x.shape[0],
        )
        return loss

    def on_validation_epoch_start(self):
        self.val_cm.zero_()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        target = self._target_to_index(y)
        logits = self(x)
        loss = self.loss_fn(logits, target)

        pred = logits.argmax(dim=1)
        self.val_cm += confusion_matrix(
            pred=pred,
            target=target,
            num_classes=int(self.hparams.num_classes),
            ignore_index=int(self.hparams.ignore_index),
        ).to(self.val_cm.device)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        metrics = metrics_from_cm(self.val_cm)
        self.log("val_miou", metrics["miou"], prog_bar=True, sync_dist=True)
        self.log("val_macro_f1", metrics["macro_f1"], prog_bar=False, sync_dist=True)
        self.log("val_pixel_acc", metrics["pixel_acc"], prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-18 + U-Net decoder baseline.")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=("phisat2", "sentinel2"),
        default="sentinel2",
        help="Dataset/datamodule to use.",
    )
    parser.add_argument("--image_dir", type=str, default=None, help="Optional override for image directory.")
    parser.add_argument("--mask_dir", type=str, default=None, help="Optional override for mask directory.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--perturbation", type=str, default="clean")
    parser.add_argument("--strength", type=float, default=0.0)

    parser.add_argument("--save_dir", type=str, default="results/baselines")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Optional Lightning precision, e.g. '16-mixed' or 'bf16-mixed'. Leave unset for default FP32.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    if args.dataset == "phisat2":
        image_dir = args.image_dir or "/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/images_phisat2_npy"
        mask_dir = args.mask_dir or "/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/masks_phisat2_npy"
        DataModuleClass = PhiSatSegDataModule
    else:
        image_dir = args.image_dir or "/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/images_s2_npy"
        mask_dir = args.mask_dir or "/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/masks_s2_npy"
        DataModuleClass = Sentinel2SegDataModule

    dm = DataModuleClass(
        image_dir=image_dir,
        mask_dir=mask_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        perturbation=args.perturbation,
        strength=args.strength,
    )
    dm.setup(stage="fit")

    in_chans = int(dm.input_shape[0])
    run_name = args.run_name or f"resnet18_unet_{args.dataset}_{args.perturbation}_s{args.strength}"

    model = ResNet18UNetLit(
        in_chans=in_chans,
        num_classes=NUM_CLASSES,
        lr=args.lr,
        focal_gamma=args.focal_gamma,
        ignore_index=IGNORE_INDEX,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(save_dir=str(save_dir), name=run_name)
    callbacks = [
        ModelCheckpoint(
            monitor="val_miou",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename="epoch={epoch:02d}-val_miou={val_miou:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer_kwargs = dict(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=args.devices if accelerator == "gpu" else 1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    if args.precision is not None:
        trainer_kwargs["precision"] = args.precision

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=dm)

    best_path = callbacks[0].best_model_path
    print(f"Best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
