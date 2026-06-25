from typing import Any
import torch
from torchmetrics import Metric
import torch.nn.functional as F

from src.metrics.stratified_metric_mixin import StratifiedMetricMixin

class DiceScoreWithStd(Metric, StratifiedMetricMixin):
    """
    Computes Dice Score (Segmentation) and its Standard Deviation.
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

        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor, label: torch.Tensor):
        if preds.ndim == 4:
            preds = torch.argmax(preds, dim=1)

        batch_size = preds.shape[0]
        preds = preds.view(batch_size, -1)
        target = target.view(batch_size, -1)

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
            cardinality = p_oh.sum(dim=1) + t_oh.sum(dim=1).float()

            epsilon = 1e-6
            dice_per_class = (2.0 * intersection) / (cardinality + epsilon)
            sample_dice = dice_per_class.mean()

            batch_scores.append(sample_dice.view(1))
            batch_labels.append(lbl.view(1))

        if batch_scores:
            self.scores.append(torch.cat(batch_scores))
            self.labels.append(torch.cat(batch_labels))

    def compute(self) -> dict[str, Any]:
        all_scores = torch.cat(self.scores) if self.scores else torch.tensor([], device=self.device)
        all_labels = torch.cat(self.labels) if self.labels else torch.tensor([], device=self.device)
        return self._compute_stratified(all_scores, all_labels)
