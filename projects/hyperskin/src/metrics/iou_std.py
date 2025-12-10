from typing import Any
import torch
from torchmetrics import Metric
import torch.nn.functional as F

from src.metrics.stratified_metric_mixin import StratifiedMetricMixin

class MeanIoUWithStd(Metric, StratifiedMetricMixin):
    """
    Computes Mean IoU (Segmentation) and its Standard Deviation.
    Also tracks the Image Classification Label to stratify results.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.include_background = include_background

        # Store scores (IoU per image) and labels (Class of the image)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor, label: torch.Tensor):
        """
        preds: (B, C, H, W) logits or (B, H, W) indices
        target: (B, H, W) indices
        label: (B,) integer labels describing the image class (e.g. Melanoma vs Nevus)
        """
        if preds.ndim == 4:
            preds = torch.argmax(preds, dim=1)

        batch_size = preds.shape[0]
        preds = preds.view(batch_size, -1)
        target = target.view(batch_size, -1)

        # Ensure label is on correct device and flattened
        if label.ndim > 1:
            label = label.view(-1)

        batch_scores = []
        batch_labels = []

        for i in range(batch_size):
            p = preds[i]
            t = target[i]
            lbl = label[i]

            p_oh = F.one_hot(p, num_classes=self.num_classes).T
            t_oh = F.one_hot(t, num_classes=self.num_classes).T

            start_idx = 0 if self.include_background else 1
            if start_idx >= self.num_classes:
                batch_scores.append(torch.tensor(0.0, device=self.device))
                batch_labels.append(lbl)
                continue

            p_oh = p_oh[start_idx:]
            t_oh = t_oh[start_idx:]

            intersection = (p_oh & t_oh).sum(dim=1).float()
            union = (p_oh | t_oh).sum(dim=1).float()

            valid_classes = union > 0

            if valid_classes.sum() == 0:
                sample_iou = torch.tensor(0.0, device=self.device)
            else:
                iou_per_class = intersection[valid_classes] / union[valid_classes]
                sample_iou = iou_per_class.mean()

            batch_scores.append(sample_iou.view(1))
            batch_labels.append(lbl.view(1))

        if batch_scores:
            self.scores.append(torch.cat(batch_scores))
            self.labels.append(torch.cat(batch_labels))

    def compute(self) -> dict[str, Any]:
        all_scores = torch.cat(self.scores) if self.scores else torch.tensor([], device=self.device)
        all_labels = torch.cat(self.labels) if self.labels else torch.tensor([], device=self.device)
        return self._compute_stratified(all_scores, all_labels)
