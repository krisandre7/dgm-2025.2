# src/models/hsi_segmentation_module.py
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.segmentation import MeanIoU, DiceScore

from src.models.smp import SMPModel
from src.utils.tags_and_run_name import add_tags_and_run_name_to_logger


class HSISegmentationModule(pl.LightningModule):
    def __init__(
        self,
        arch_name: str = "unet",
        encoder_name: str = "resnet18",
        num_classes: int = 2,
        in_chans: int = 16,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        lr: float = 1e-4,
        loss_name: str = "ce",  # could be "ce" or "dice" or "ce+dice"
    ):
        super().__init__()
        self.save_hyperparameters()

        # 🔹 Build Segmentation Model
        self.net = SMPModel(
            arch_name=self.hparams.arch_name,
            encoder_name=self.hparams.encoder_name,
            num_classes=self.hparams.num_classes,
            in_chans=self.hparams.in_chans,
            pretrained=self.hparams.pretrained,
            freeze_encoder=self.hparams.freeze_encoder,
        )

        # 🔹 Loss
        if loss_name == "ce":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == "dice":
            import segmentation_models_pytorch as smp

            self.criterion = smp.losses.DiceLoss(mode="multiclass")
        elif loss_name == "ce+dice":
            import segmentation_models_pytorch as smp

            self.criterion = (
                nn.CrossEntropyLoss(),
                smp.losses.DiceLoss(mode="multiclass"),
            )
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        # 🔹 Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_iou = MeanIoU(num_classes=num_classes)
        self.val_iou = MeanIoU(num_classes=num_classes)
        self.test_iou = MeanIoU(num_classes=num_classes)

        self.train_dice = DiceScore(num_classes=num_classes)
        self.val_dice = DiceScore(num_classes=num_classes)
        self.test_dice = DiceScore(num_classes=num_classes)

        # Track best val IoU
        self.val_iou_best = MaxMetric()
        self.val_dice_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def _shared_step(self, batch: Any):
        x = batch["image"]  # shape: (B, C, H, W)
        y = batch["mask"]   # shape: (B, H, W)
        label = batch['label']
        logits = self.forward(x)  # shape: (B, num_classes, H, W)

        if isinstance(self.criterion, tuple):  # ce+dice combo
            loss = self.criterion[0](logits, y) + self.criterion[1](logits, y)
        else:
            loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, label

    # ---------------- TRAINING ----------------
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, label = self._shared_step(batch)
        self.train_loss(loss)
        self.train_iou(preds, targets)
        self.train_dice(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True)
        return loss

    # ---------------- VALIDATION ----------------
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, label = self._shared_step(batch)
        self.val_loss(loss)
        self.val_iou(preds, targets)
        self.val_dice(preds, targets)

        self.log("val/loss", self.val_loss, on_epoch=True)
        self.log("val/iou", self.val_iou, on_epoch=True)
        self.log("val/dice", self.val_dice, on_epoch=True)

    def on_validation_epoch_end(self):
        current_iou = self.val_iou.compute()
        self.val_iou_best(current_iou)
        self.log("val/iou_best", self.val_iou_best.compute())

        current_dice = self.val_dice.compute()
        self.val_dice_best(current_dice)
        self.log("val/dice_best", self.val_dice_best.compute())

    # ---------------- TEST ----------------
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, label = self._shared_step(batch)
        self.test_loss(loss)
        self.test_iou(preds, targets)
        self.test_dice(preds, targets)

        self.log("test/loss", self.test_loss, on_epoch=True)
        self.log("test/iou", self.test_iou, on_epoch=True)
        self.log("test/dice", self.test_dice, on_epoch=True)

    # ---------------- OPTIMIZER ----------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def _get_tags_and_run_name(self):
        """
        Automatically derive tags and a run name from HSISegmentationModule hyperparameters.
        Useful for tracking and organizing experiments in W&B.
        """
        hparams = getattr(self, "hparams", None)
        if hparams is None:
            return [], ""

        tags = []
        run_name = ""

        # Architecture and encoder info
        tags.append(hparams.arch_name)
        tags.append(hparams.encoder_name)
        run_name += f"{hparams.arch_name}_{hparams.encoder_name}"

        # Pretrained tag
        if hparams.pretrained:
            tags.append("pretrained")
            run_name += "_pt"
        else:
            tags.append("not_pretrained")

        # Freeze encoder
        if hparams.freeze_encoder:
            tags.append("frozen_encoder")
            run_name += "_fe"

        # Loss info
        loss_tag = hparams.loss_name.lower().replace("+", "_")
        tags.append(loss_tag)
        run_name += f"_{loss_tag}"

        # Channels and number of classes
        tags.append(f"in{hparams.in_chans}")
        tags.append(f"classes{hparams.num_classes}")
        run_name += f"_in{hparams.in_chans}_c{hparams.num_classes}"

        return tags, run_name

    def setup(self, stage: str) -> None:
        add_tags_and_run_name_to_logger(self)
