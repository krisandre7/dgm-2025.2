# src/models/hsi_segmentation_module.py
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric, MaxMetric, Metric

from src.metrics.dice_std import DiceScoreWithStd
from src.metrics.iou_std import MeanIoUWithStd
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
        loss_name: str = "ce",
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

        # Custom metrics that track score per classification label
        self.train_iou = MeanIoUWithStd(num_classes=num_classes)
        self.val_iou = MeanIoUWithStd(num_classes=num_classes)
        self.test_iou = MeanIoUWithStd(num_classes=num_classes)

        self.train_dice = DiceScoreWithStd(num_classes=num_classes)
        self.val_dice = DiceScoreWithStd(num_classes=num_classes)
        self.test_dice = DiceScoreWithStd(num_classes=num_classes)

        # Track best val IoU/Dice (Overall Mean)
        self.val_iou_best = MaxMetric()
        self.val_dice_best = MaxMetric()

        # Cache for id to string map
        self._id_to_name = None

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def _get_id_to_name_map(self) -> dict[int, str]:
        """Lazy loader for label map from DataModule."""
        if self._id_to_name is None:
            if self.trainer and hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                # The DataModule has get_labels_map() -> {'name': id}
                labels_map = self.trainer.datamodule.get_labels_map()
                self._id_to_name = {v: k for k, v in labels_map.items()}
            else:
                # Fallback if no datamodule attached yet
                return {}
        return self._id_to_name

    def _shared_step(self, batch: Any):
        x = batch["image"]
        y = batch["mask"]
        label = batch["label"]  # This is the image-level class (int)

        logits = self.forward(x)

        if isinstance(self.criterion, tuple):
            loss = self.criterion[0](logits, y) + self.criterion[1](logits, y)
        else:
            loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, label

    def _log_stratified_metric(self, split: str, metric_name: str, metric_result: dict[str, Any]):
        """
        Helper to log the dictionary structure returned by StratifiedMetrics.
        metric_result looks like: {'overall': {'mean': ..., 'std': ...}, 'per_class': {0: {...}, 1: {...}}}
        """
        # 1. Log Overall
        self.log(f"{split}/{metric_name}", metric_result["overall"]["mean"])
        if split != "train":  # usually noise in train std
            self.log(f"{split}/{metric_name}_std", metric_result["overall"]["std"])

        # 2. Log Per Class
        id_to_name = self._get_id_to_name_map()

        for class_id, stats in metric_result.get("per_class", {}).items():
            # Get string name if available, else use "class_X"
            class_str = id_to_name.get(class_id, f"class_{class_id}")
            class_str = class_str.lower().replace(" ", "_")

            self.log(f"{split}/{metric_name}_{class_str}", stats["mean"])
            if split != "train":
                self.log(f"{split}/{metric_name}_{class_str}_std", stats["std"])

    # ---------------- TRAINING ----------------
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, label = self._shared_step(batch)
        self.train_loss(loss)

        # Pass label for stratification
        self.train_iou(preds, targets, label)
        self.train_dice(preds, targets, label)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # Retrieve computed dictionaries
        iou_res = self.train_iou.compute()
        dice_res = self.train_dice.compute()

        # Helper to log mean and per-class
        self._log_stratified_metric("train", "iou", iou_res)
        self._log_stratified_metric("train", "dice", dice_res)

        self.train_iou.reset()
        self.train_dice.reset()

    # ---------------- VALIDATION ----------------
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, label = self._shared_step(batch)
        self.val_loss(loss)
        self.val_iou(preds, targets, label)
        self.val_dice(preds, targets, label)
        self.log("val/loss", self.val_loss, on_epoch=True)

    def on_validation_epoch_end(self):
        iou_res = self.val_iou.compute()
        dice_res = self.val_dice.compute()

        # Log detailed metrics
        self._log_stratified_metric("val", "iou", iou_res)
        self._log_stratified_metric("val", "dice", dice_res)

        # Track Best (based on global mean)
        current_iou_mean = iou_res["overall"]["mean"]
        current_dice_mean = dice_res["overall"]["mean"]

        self.val_iou_best(current_iou_mean)
        self.val_dice_best(current_dice_mean)

        self.log("val/iou_best", self.val_iou_best.compute())
        self.log("val/dice_best", self.val_dice_best.compute())

        self.val_iou.reset()
        self.val_dice.reset()

    # ---------------- TEST ----------------
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, label = self._shared_step(batch)
        self.test_loss(loss)
        self.test_iou(preds, targets, label)
        self.test_dice(preds, targets, label)
        self.log("test/loss", self.test_loss, on_epoch=True)

    def on_test_epoch_end(self):
        iou_res = self.test_iou.compute()
        dice_res = self.test_dice.compute()

        self._log_stratified_metric("test", "iou", iou_res)
        self._log_stratified_metric("test", "dice", dice_res)

        self.test_iou.reset()
        self.test_dice.reset()

    # ---------------- OPTIMIZER ----------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def _get_tags_and_run_name(self):
        hparams = getattr(self, "hparams", None)
        if hparams is None:
            return [], ""

        tags = []
        run_name = ""

        tags.append(hparams.arch_name)
        tags.append(hparams.encoder_name)
        run_name += f"{hparams.arch_name}_{hparams.encoder_name}"

        if hparams.pretrained:
            tags.append("pretrained")
            run_name += "_pt"
        else:
            tags.append("not_pretrained")

        if hparams.freeze_encoder:
            tags.append("frozen_encoder")
            run_name += "_fe"

        loss_tag = hparams.loss_name.lower().replace("+", "_")
        tags.append(loss_tag)
        run_name += f"_{loss_tag}"

        tags.append(f"in{hparams.in_chans}")
        tags.append(f"classes{hparams.num_classes}")
        run_name += f"_in{hparams.in_chans}_c{hparams.num_classes}"

        return tags, run_name

    def setup(self, stage: str) -> None:
        add_tags_and_run_name_to_logger(self)
