import torch


class StratifiedMetricMixin:
    """Helper to organize computation of mean/std per group (label)."""

    def _compute_stratified(
        self, all_scores: torch.Tensor, all_labels: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Returns:
            {
                "overall": {"mean": ..., "std": ...},
                "per_class": {
                    0: {"mean": ..., "std": ...},
                    1: {"mean": ..., "std": ...},
                    ...
                }
            }
        """
        results = {}

        # 1. Overall Statistics
        if all_scores.numel() > 0:
            results["overall"] = {
                "mean": all_scores.mean(),
                "std": all_scores.std() if all_scores.numel() > 1 else torch.tensor(0.0, device=all_scores.device),
            }
        else:
            results["overall"] = {
                "mean": torch.tensor(0.0, device=self.device),
                "std": torch.tensor(0.0, device=self.device),
            }

        # 2. Per-Class Statistics
        results["per_class"] = {}
        if all_labels.numel() > 0:
            unique_labels = torch.unique(all_labels)
            for lbl in unique_labels:
                lbl_item = lbl.item()
                mask = all_labels == lbl
                scores_lbl = all_scores[mask]

                if scores_lbl.numel() > 0:
                    results["per_class"][lbl_item] = {
                        "mean": scores_lbl.mean(),
                        "std": scores_lbl.std() if scores_lbl.numel() > 1 else torch.tensor(0.0, device=self.device),
                    }

        return results
