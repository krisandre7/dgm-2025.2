import numpy as np
from typing import Dict


class SIDMetric:
    """
    Spectral Information Divergence (SID) computed from MeanSpectraMetric outputs.

    This metric compares REAL vs FAKE class-conditional mean spectra.
    Each mean spectrum is treated as a discrete probability distribution
    over spectral bands after normalization.

    Expected input format (from MeanSpectraMetric.compute()):

    {
        "real": {
            "<class_name>": {"mean": np.ndarray, "std": np.ndarray},
            ...
        },
        "fake": {
            "<class_name>": {"mean": np.ndarray, "std": np.ndarray},
            ...
        }
    }
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def _normalize_spectrum(self, s: np.ndarray) -> np.ndarray:
        """
        Convert a spectrum into a probability distribution over bands.
        """
        s = np.clip(s, self.eps, None)
        return s / s.sum()

    def _sid(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute symmetric SID between two normalized spectra.
        """
        return 0.5 * (
            np.sum(p * np.log(p / q)) +
            np.sum(q * np.log(q / p))
        )

    def compute(
        self,
        mean_spectra_stats: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> Dict[str, float]:
        """
        Compute SID per class between real and fake mean spectra.

        Returns:
            Dict[str, float]:
                {
                    "normal_skin": SID_value,
                    "lesion_0": SID_value,
                    "lesion_1": SID_value,
                    ...
                }
        """
        real_stats = mean_spectra_stats.get("real", {})
        fake_stats = mean_spectra_stats.get("fake", {})

        sid_results = {}

        for cls_name in real_stats.keys():
            if cls_name not in fake_stats:
                # Cannot compare if fake class is missing
                continue

            mean_real = real_stats[cls_name]["mean"]
            mean_fake = fake_stats[cls_name]["mean"]

            p = self._normalize_spectrum(mean_real)
            q = self._normalize_spectrum(mean_fake)

            sid_results[cls_name] = float(self._sid(p, q))

        return sid_results
