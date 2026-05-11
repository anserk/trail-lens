# keep class balance
# avoid duplicate/source leakage
# split by source website
# make train/val/test
# save split indexes to disk


import random

from torch import Tensor
from torch.utils.data import Dataset

from trail_lens.dataset import TrailLensDataset


class TrailLensDataSubset(Dataset):
    def __init__(self, dataset: TrailLensDataset, indexes: list[int]) -> None:
        self.dataset = dataset
        self.indexes = indexes

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.dataset[self.indexes[index]]


def train_val_split(
    dataset: TrailLensDataset, val_fraction: float, seed: int
) -> tuple[TrailLensDataSubset, TrailLensDataSubset]:

    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError(f"val_fraction needs to be between 0 and 1, got {val_fraction}")

    len_dataset = len(dataset)
    indexes = list(range(len_dataset))
    rng = random.Random(seed)
    rng.shuffle(indexes)

    validation_count = int(len_dataset * val_fraction)

    validation_indexes = indexes[:validation_count]
    train_indexes = indexes[validation_count:]

    return TrailLensDataSubset(dataset, train_indexes), TrailLensDataSubset(
        dataset, validation_indexes
    )
