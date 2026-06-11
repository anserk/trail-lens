import random
from collections.abc import Callable, Sized
from dataclasses import dataclass

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


@dataclass
class SplitIndexes:
    train: list[int]
    val: list[int]
    test: list[int]


def make_splits(
    dataset: Sized, val_fraction: float, test_fraction: float, seed: int
) -> SplitIndexes:
    """
    Split an array of indexes in 3 different arrays:
        - train, used for training the model;
        - val, used for validating progress during training epochs;
        - test, used for final model evaluation.

    Args:
        dataset: Array of indexes to split.
        val_fraction: .
        test_fraction:
        seed:

    Returns:
        A class containing arrays split in train, val, test.

    Raises:
        ValueError: If fraction splits are invalid.

    """

    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError(f"val_fraction needs to be between 0 and 1, got {val_fraction}")
    if test_fraction <= 0 or test_fraction >= 1:
        raise ValueError(f"test_fraction needs to be between 0 and 1, got {test_fraction}")

    if val_fraction + test_fraction >= 1:
        raise ValueError(
            f"val_fraction + test_fraction needs to be less than 1, got {val_fraction + test_fraction}"
        )

    len_dataset = len(dataset)
    indexes = list(range(len_dataset))
    rng = random.Random(seed)
    rng.shuffle(indexes)

    validation_count = int(len_dataset * val_fraction)
    test_count = int(len_dataset * test_fraction)

    validation_indexes = indexes[:validation_count]
    test_indexes = indexes[validation_count : validation_count + test_count]
    train_indexes = indexes[validation_count + test_count :]

    return SplitIndexes(train=train_indexes, val=validation_indexes, test=test_indexes)
