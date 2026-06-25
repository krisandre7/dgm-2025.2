import numpy as np
import torch
from torch.utils.data import Sampler


class FiniteSampler(Sampler[int]):
    """Samples a fixed number of elements from the dataset.

    If more samples are requested than the dataset contains, the sampler keeps
    yielding indices with replacement only when allow_replacement=True. This is
    useful for prediction-time unconditional generators, where dataset rows only
    provide batch sizing.
    """

    def __init__(self, data_source, num_samples, allow_replacement=False):
        self.data_source = data_source
        self.num_samples = num_samples if allow_replacement else min(num_samples, len(data_source))
        self.allow_replacement = allow_replacement

    def __iter__(self):
        dataset_size = len(self.data_source)
        if dataset_size == 0:
            return iter([])

        if self.num_samples <= dataset_size or not self.allow_replacement:
            indices = np.random.permutation(dataset_size)[: self.num_samples]
        else:
            indices = np.concatenate(
                [
                    np.random.permutation(dataset_size),
                    np.random.choice(
                        dataset_size,
                        size=self.num_samples - dataset_size,
                        replace=True,
                    ),
                ]
            )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
