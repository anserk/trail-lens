# keep class balance
# avoid duplicate/source leakage
# split by source website
# make train/val/test
# save split indexes to disk


import random
from collections.abc import Callable

from torch import Tensor
from torch.utils.data import Dataset

from trail_lens.dataset import TrailLensDataset


class TrailLensDataSubset(Dataset):
    def __init__(
        self,
        dataset: TrailLensDataset,
        indexes: list[int],
        transform: Callable[[Tensor], Tensor] | None = None,
        target_transform: Callable[[int], int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.indexes = indexes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        tensor, label = self.dataset[self.indexes[index]]
        if self.transform:
            tensor = self.transform(tensor)
        if self.target_transform:
            label = self.target_transform(label)

        return tensor, label


def train_val_split(
    dataset: TrailLensDataset, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:

    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError(f"val_fraction needs to be between 0 and 1, got {val_fraction}")

    len_dataset = len(dataset)
    indexes = list(range(len_dataset))
    rng = random.Random(seed)
    rng.shuffle(indexes)

    validation_count = int(len_dataset * val_fraction)

    validation_indexes = indexes[:validation_count]
    train_indexes = indexes[validation_count:]

    return train_indexes, validation_indexes
