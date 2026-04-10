import torch
import torch.nn as nn
import pytorch_lightning as pl

from .losses import FocalLoss

class SegmentationTask(pl.LightningModule):
    def __init__(self, model, use_focal=False, alpha=1, gamma=2, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        if use_focal:
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.ndim == 4:
            y = torch.argmax(y, dim=1)

        logits = self.model(x)
        loss = self.loss_fn(logits, y.long())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)