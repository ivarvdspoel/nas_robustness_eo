#!/usr/bin/env python3
"""
Train a ViT-S/16 + U-Net-style decoder semantic-segmentation baseline on either
PhiSat-2 or Sentinel-2 patches.

Expected dataloader:
    from src.data_loader.data_loader import PhiSatSegDataModule, Sentinel2SegDataModule

Default dataset paths follow the NAS main.py setup:
    PhiSat-2:   8 x 256 x 256 images, 4 x 256 x 256 one-hot masks
    Sentinel-2: 7 x 128 x 128 images, 4 x 128 x 128 one-hot masks

Model:
    timm ViT-S/16 encoder, pretrained=False
    U-Net-style dense decoder using intermediate ViT block features as skip features

Training:
    Adam, lr=1e-3, focal loss, 10 epochs by default
"""
from __future__ import annotations

import argparse
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
# ViT-S/16 U-Net-style model
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


class ViTS16FeatureExtractor(nn.Module):
    """Extract intermediate spatial feature maps from a timm ViT-S/16 backbone.

    A ViT does not naturally produce a CNN-like feature pyramid. Here, we take
    intermediate transformer block outputs and reshape the patch tokens back to
    their spatial token grid. The decoder then turns these same-resolution token
    grids into a dense U-Net-style segmentation output.
    """

    def __init__(
        self,
        in_chans: int,
        img_size: Tuple[int, int],
        patch_size: int = 16,
        out_indices: Sequence[int] = (2, 5, 8, 11),
    ):
        super().__init__()
        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0:
            raise ValueError(
                f"ViT-S/16 requires image height and width divisible by {patch_size}. "
                f"Got img_size={img_size}."
            )

        self.img_size = tuple(img_size)
        self.patch_size = int(patch_size)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.out_indices = tuple(out_indices)

        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,
        )
        self.embed_dim = int(getattr(self.backbone, "embed_dim", 384))
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

        max_idx = max(self.out_indices)
        if max_idx >= len(self.backbone.blocks):
            raise ValueError(
                f"Requested out_indices={self.out_indices}, but backbone only has "
                f"{len(self.backbone.blocks)} transformer blocks."
            )

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x[:, self.num_prefix_tokens:, :]
        b, n, c = tokens.shape
        gh, gw = self.grid_size
        expected_n = gh * gw
        if n != expected_n:
            raise RuntimeError(
                f"Unexpected number of ViT patch tokens: got {n}, expected {expected_n}. "
                "Check img_size and patch_size."
            )
        return tokens.transpose(1, 2).reshape(b, c, gh, gw).contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Patchify image to token sequence.
        x = self.backbone.patch_embed(x)

        # Add class token and positional embedding using timm's own logic where available.
        if hasattr(self.backbone, "_pos_embed"):
            x = self.backbone._pos_embed(x)
        else:
            cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.backbone.pos_embed
            x = self.backbone.pos_drop(x)

        if hasattr(self.backbone, "patch_drop"):
            x = self.backbone.patch_drop(x)
        if hasattr(self.backbone, "norm_pre"):
            x = self.backbone.norm_pre(x)

        features: list[torch.Tensor] = []
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i in self.out_indices:
                features.append(self._tokens_to_map(x))

        return features


class ViTS16UNet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        img_size: Tuple[int, int],
        num_classes: int = NUM_CLASSES,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
    ):
        super().__init__()
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four values, e.g. (256, 128, 64, 32).")

        self.encoder = ViTS16FeatureExtractor(
            in_chans=in_chans,
            img_size=img_size,
            patch_size=16,
            out_indices=(2, 5, 8, 11),
        )
        embed_dim = self.encoder.embed_dim

        # Deep token map at 1/16 resolution.
        self.bottleneck = ConvBlock(embed_dim, decoder_channels[0])

        # ViT feature maps all start at 1/16 resolution. These projections turn
        # earlier/middle block outputs into pseudo-skip features at 1/8, 1/4, and
        # 1/2 resolution for a dense U-Net-style decoder.
        self.skip_1_8 = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True),
        )
        self.skip_1_4 = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True),
        )
        self.skip_1_2 = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )

        self.up_1_8 = UpBlock(decoder_channels[0], decoder_channels[1], decoder_channels[1])
        self.up_1_4 = UpBlock(decoder_channels[1], decoder_channels[2], decoder_channels[2])
        self.up_1_2 = UpBlock(decoder_channels[2], decoder_channels[3], decoder_channels[3])

        self.final_conv = ConvBlock(decoder_channels[3], decoder_channels[3])
        self.seg_head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        f_early, f_mid, f_late, f_deep = self.encoder(x)

        y = self.bottleneck(f_deep)

        skip_1_8 = self.skip_1_8(f_late)
        skip_1_8 = F.interpolate(skip_1_8, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.up_1_8(y, skip_1_8)

        skip_1_4 = self.skip_1_4(f_mid)
        skip_1_4 = F.interpolate(skip_1_4, scale_factor=4, mode="bilinear", align_corners=False)
        y = self.up_1_4(y, skip_1_4)

        skip_1_2 = self.skip_1_2(f_early)
        skip_1_2 = F.interpolate(skip_1_2, scale_factor=8, mode="bilinear", align_corners=False)
        y = self.up_1_2(y, skip_1_2)

        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        y = self.final_conv(y)
        return self.seg_head(y)


# -----------------------------------------------------------------------------
# Lightning module
# -----------------------------------------------------------------------------

class ViTS16UNetLit(pl.LightningModule):
    def __init__(
        self,
        in_chans: int,
        img_size: Tuple[int, int],
        num_classes: int = NUM_CLASSES,
        lr: float = 1e-3,
        focal_gamma: float = 2.0,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ViTS16UNet(
            in_chans=in_chans,
            img_size=img_size,
            num_classes=num_classes,
        )
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
    parser = argparse.ArgumentParser(description="Train ViT-S/16 + U-Net-style decoder baseline.")

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

    input_shape = tuple(dm.input_shape)
    in_chans = int(input_shape[0])
    img_size = (int(input_shape[1]), int(input_shape[2]))

    run_name = args.run_name or f"vit_s16_unet_{args.dataset}_{args.perturbation}_s{args.strength}"

    model = ViTS16UNetLit(
        in_chans=in_chans,
        img_size=img_size,
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
