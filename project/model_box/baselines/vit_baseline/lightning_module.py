import torch
import torch.nn as nn
import pytorch_lightning as pl

from .losses import *

class SegmentationTask(pl.LightningModule):
    def __init__(self, model, use_focal=False, alpha=1, gamma=2, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr

        if use_focal:
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def _prepare_targets(self, y):
        # If masks are one-hot encoded: [B, C, H, W] -> [B, H, W]
        if y.ndim == 4:
            y = torch.argmax(y, dim=1)
        return y.long()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = self._prepare_targets(y)

        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = self._prepare_targets(y)

        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)