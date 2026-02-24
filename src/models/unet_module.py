"""
UNet++ Lightning Module for wildfire burned-area segmentation.

Adapted from the ORORA project (Orion AI Lab, National Observatory of Athens).
"""

from typing import Any
import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from torchmetrics import AveragePrecision
from transformers import get_cosine_schedule_with_warmup


class plUNET(pl.LightningModule):
    """
    UNet++ with a pre-trained encoder for binary burned-area classification.

    Inputs: local patch (C×H×W) + positional encoding (4×H×W)
    Output: 2-class segmentation map (no-fire vs fire)
    """

    def __init__(
        self,
        input_vars: list = None,
        positional_vars: list = None,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        loss: str = "ce",
        encoder: str = "efficientnet-b1",
        warmup_ratio: float = 0.05,
        sea_masked: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        in_channels = len(input_vars or []) + len(positional_vars or [])
        self.net = smp.UnetPlusPlus(
            encoder_name=encoder, in_channels=in_channels, classes=2
        )

        if loss == "dice":
            self.criterion = smp.losses.DiceLoss(mode="multiclass")
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.sea_masked = sea_masked
        ignore_idx = 2 if sea_masked else None
        self.val_auprc = AveragePrecision(
            task="binary", num_classes=1, ignore_index=ignore_idx
        )
        self.test_auprc = AveragePrecision(
            task="binary", num_classes=1, ignore_index=ignore_idx
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x_local = batch["x_local"].squeeze().float()
        x_local_pos = batch["x_local_pos"].float()
        x = torch.cat([x_local, x_local_pos], dim=1)
        y = batch["y_local"].long()

        # Pad to be divisible by 32 (required by encoder)
        pad_size = (x_local.shape[2] % 32) // 2
        if pad_size > 0:
            x = torch.nn.functional.pad(
                x, (pad_size, pad_size, pad_size, pad_size), mode="constant", value=0
            )
            y = torch.nn.functional.pad(
                y, (pad_size, pad_size, pad_size, pad_size), mode="constant", value=0
            )

        logits = self.forward(x.float())
        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        self.val_auprc.update(preds.flatten(), targets.flatten())
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)
        self.test_auprc.update(preds.flatten(), targets.flatten())
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        num_epochs = self.trainer.max_epochs
        nbatches = len(self.trainer.datamodule.train_dataloader())
        total_steps = num_epochs * nbatches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            "monitor": "train/loss",
        }
