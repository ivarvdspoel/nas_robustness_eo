#!/usr/bin/env python3
"""
Train a TerraMind + U-Net decoder semantic-segmentation baseline on Sentinel-2
patches only.

Expected dataloader:
    from src.data_loader.data_loader import Sentinel2SegDataModule

Default dataset paths follow the NAS main.py setup:
    Sentinel-2: 7 x 128 x 128 images, 4 x 128 x 128 one-hot masks

Model:
    TerraMind backbone through TerraTorch, pretrained=True by default
    modality subset: S2L1C only
    band subset: Sentinel-2 B02--B08 equivalent channels only
    decoder: TerraTorch UNetDecoder

Training:
    Adam, lr=1e-3, focal loss, 10 epochs by default

Notes:
    TerraMind is a transformer-like geospatial foundation model, so it does not
    emit a native CNN-style feature pyramid. The TerraTorch necks below select
    intermediate transformer layers, reshape tokens into spatial maps, then learn
    a pyramidal interpolation so the UNetDecoder receives multi-scale features.

    The default band names below follow TerraTorch/TerraMind semantic naming for
    a Sentinel-2 L1C subset in the same order as the project dataloader:
        B02, B03, B04, B05, B06, B07, B08
    If your installed TerraTorch version expects literal band IDs instead, run:
        --s2_bands B02,B03,B04,B05,B06,B07,B08
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import lightning as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

from src.data_loader.data_loader import Sentinel2SegDataModule


NUM_CLASSES = 4
IGNORE_INDEX = 255
MODALITY = "S2L1C"

# Project Sentinel-2 dataloader channel order: [B02, B03, B04, B05, B06, B07, B08].
# TerraMind/TerraTorch semantic names for the same subset.
DEFAULT_S2L1C_BANDS = [
    "BLUE",        # B02
    "GREEN",       # B03
    "RED",         # B04
    "RED_EDGE_1",  # B05
    "RED_EDGE_2",  # B06
    "RED_EDGE_3",  # B07
    "NIR_BROAD",   # B08
]


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


def unwrap_model_output(out: Any) -> torch.Tensor:
    """Return logits from common TerraTorch/PyTorch output containers."""
    if torch.is_tensor(out):
        return out

    if isinstance(out, dict):
        for key in ("output", "logits", "out", "prediction", "predictions", "mask", "masks"):
            if key in out:
                return unwrap_model_output(out[key])
        for value in out.values():
            try:
                return unwrap_model_output(value)
            except TypeError:
                pass

    if isinstance(out, (list, tuple)):
        for value in out:
            try:
                return unwrap_model_output(value)
            except TypeError:
                pass

    if hasattr(out, "__dict__"):
        d = vars(out)
        for key in ("output", "logits", "out", "prediction", "predictions", "mask", "masks"):
            if key in d:
                return unwrap_model_output(d[key])
        for value in d.values():
            try:
                return unwrap_model_output(value)
            except TypeError:
                pass

    for attr in ("output", "logits", "out", "prediction", "predictions", "mask", "masks"):
        if hasattr(out, attr):
            try:
                return unwrap_model_output(getattr(out, attr))
            except TypeError:
                pass

    if hasattr(out, "to_tuple"):
        return unwrap_model_output(out.to_tuple())

    raise TypeError(
        f"Could not unwrap model output of type {type(out)}. "
        f"Available attrs: {[a for a in dir(out) if not a.startswith('_')][:50]}"
    )


# -----------------------------------------------------------------------------
# TerraMind + UNetDecoder model
# -----------------------------------------------------------------------------

class TerraMindS2UNet(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone: str = "terramind_v1_base",
        pretrained: bool = True,
        bands: Sequence[str] = DEFAULT_S2L1C_BANDS,
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
        head_dropout: float = 0.1,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ):
        super().__init__()
        try:
            from terratorch.tasks import SemanticSegmentationTask
        except Exception as exc:
            raise ImportError(
                "TerraMind baseline requires TerraTorch. Install with: pip install 'terratorch>=1.2.4'"
            ) from exc

        # TerraMind tiny/small/base have 12 transformer blocks. These are the
        # standard layer indices used in TerraTorch configs for tiny/small/base.
        # For terramind_v1_large, use --select_indices 5,11,17,23.
        task = SemanticSegmentationTask(
            model_factory="EncoderDecoderFactory",
            model_args={
                "backbone": backbone,
                "backbone_pretrained": pretrained,
                "backbone_modalities": [MODALITY],
                "backbone_bands": {MODALITY: list(bands)},
                "backbone_merge_method": "mean",
                "necks": [
                    {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                    {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                    {"name": "LearnedInterpolateToPyramidal"},
                ],
                "decoder": "UNetDecoder",
                "decoder_channels": list(decoder_channels),
                "head_dropout": head_dropout,
                "num_classes": num_classes,
            },
            loss="ce",
            optimizer="AdamW",
            lr=1e-3,
            ignore_index=IGNORE_INDEX,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
        )
        self.model = task.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TerraTorch models should receive a modality dictionary for TerraMind.
        # A fallback is kept for version compatibility.
        try:
            out = self.model({MODALITY: x})
        except Exception:
            out = self.model(x)
        return unwrap_model_output(out)


# -----------------------------------------------------------------------------
# Lightning module
# -----------------------------------------------------------------------------

class TerraMindS2UNetLit(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        lr: float = 1e-3,
        focal_gamma: float = 2.0,
        ignore_index: int = IGNORE_INDEX,
        backbone: str = "terramind_v1_base",
        pretrained: bool = True,
        bands: Sequence[str] = DEFAULT_S2L1C_BANDS,
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
        head_dropout: float = 0.1,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        resize_to: int = 224,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["bands", "decoder_channels"])
        self.hparams.bands = list(bands)
        self.hparams.decoder_channels = list(decoder_channels)

        self.model = TerraMindS2UNet(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            bands=bands,
            decoder_channels=decoder_channels,
            head_dropout=head_dropout,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
        )
        self.loss_fn = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)

        self.register_buffer(
            "val_cm",
            torch.zeros(num_classes, num_classes, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]

        resize_to = int(self.hparams.resize_to)
        if resize_to > 0 and original_size != (resize_to, resize_to):
            x_model = F.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
        else:
            x_model = x

        logits = self.model(x_model)

        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        return logits

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
        if x.shape[1] != 7:
            raise ValueError(
                f"TerraMind S2 baseline expects 7 Sentinel-2 channels [B02..B08], got {tuple(x.shape)}."
            )
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
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=float(self.hparams.lr))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_csv_list(value: str, cast=str) -> list:
    if value is None or value == "":
        return []
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TerraMind + UNetDecoder baseline on Sentinel-2 only.")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=("sentinel2",),
        default="sentinel2",
        help="Only Sentinel-2 is supported here. PhiSat-2 should later use PhiSatNet.",
    )
    parser.add_argument("--image_dir", type=str, default=None, help="Optional override for Sentinel-2 image directory.")
    parser.add_argument("--mask_dir", type=str, default=None, help="Optional override for Sentinel-2 mask directory.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--perturbation", type=str, default="clean")
    parser.add_argument("--strength", type=float, default=0.0)

    parser.add_argument("--backbone", type=str, default="terramind_v1_base")
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable TerraMind pretrained weights. Default is pretrained=True because TerraMind is the foundation-model baseline.",
    )
    parser.add_argument(
        "--s2_bands",
        type=str,
        default=",".join(DEFAULT_S2L1C_BANDS),
        help=(
            "Comma-separated TerraMind/TerraTorch band subset, matching the dataloader channel order. "
            "Default maps B02--B08 to semantic names. If your version expects literal IDs, "
            "use B02,B03,B04,B05,B06,B07,B08."
        ),
    )
    parser.add_argument(
        "--decoder_channels",
        type=str,
        default="512,256,128,64",
        help="Comma-separated UNetDecoder channels. Default is suitable for TerraMind base.",
    )
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument(
        "--resize_to",
        type=int,
        default=224,
        help="Resize inputs before TerraMind, then resize logits back. Use 0 to preserve native 128x128.",
    )

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

    image_dir = args.image_dir or "/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/images_s2_npy"
    mask_dir = args.mask_dir or "/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/masks_s2_npy"

    dm = Sentinel2SegDataModule(
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
    if int(input_shape[0]) != 7:
        raise ValueError(
            f"TerraMind S2 baseline expects Sentinel-2 input shape [7,H,W], got {input_shape}."
        )

    bands = parse_csv_list(args.s2_bands, str)
    decoder_channels = parse_csv_list(args.decoder_channels, int)

    if len(bands) != 7:
        raise ValueError(
            f"Expected exactly 7 S2 band names matching [B02..B08], got {len(bands)}: {bands}"
        )

    run_name = args.run_name or f"terramind_s2_unet_{args.perturbation}_s{args.strength}"

    model = TerraMindS2UNetLit(
        num_classes=NUM_CLASSES,
        lr=args.lr,
        focal_gamma=args.focal_gamma,
        ignore_index=IGNORE_INDEX,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        bands=bands,
        decoder_channels=decoder_channels,
        head_dropout=args.head_dropout,
        freeze_backbone=args.freeze_backbone,
        freeze_decoder=args.freeze_decoder,
        resize_to=args.resize_to,
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

    print("Training TerraMind Sentinel-2 baseline")
    print(f"  backbone: {args.backbone}")
    print(f"  pretrained: {not args.no_pretrained}")
    print(f"  modality: {MODALITY}")
    print(f"  bands: {bands}")
    print(f"  decoder: UNetDecoder, channels={decoder_channels}")
    print(f"  resize_to: {args.resize_to if args.resize_to > 0 else 'native'}")

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=dm)

    best_path = callbacks[0].best_model_path
    print(f"Best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
