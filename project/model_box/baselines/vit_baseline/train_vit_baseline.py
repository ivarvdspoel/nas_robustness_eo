import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from dataset_box.data_loader import SegmentationDataModule
from model_box.baselines.vit_baseline.vit_seg import ViTSegmentation
from model_box.baselines.vit_baseline.lightning_module import SegmentationTask

import torch
from pathlib import Path

def main():
    root_dir = "/tmp/ivanderspoel/burn_dataset"
    batch_size = 8
    num_workers = 2
    val_split = 0.3

    datamodule = SegmentationDataModule(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
    )


    num_classes = 4
    img_size = 256
    in_chans = 7

    model = ViTSegmentation(
        num_classes=num_classes,
        img_size=img_size,
        pretrained=False,
        backbone_name="vit_small_patch16_224",
    )

    task = SegmentationTask(
        model=model,
        use_focal=False,
        lr=1e-4,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="vit_s16_256_{epoch:02d}_{val_loss:.4f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(task, datamodule=datamodule)

    Path("model_box/saved_models").mkdir(parents=True, exist_ok=True)

    torch.save(
        task.model.state_dict(),
        "model_box/saved_models/vit_20epochs.pt"
    )


if __name__ == "__main__":
    main()