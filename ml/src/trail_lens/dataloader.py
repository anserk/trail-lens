# [batch, channels, height, width]

# Dataset:
#   knows how to get one sample

# DataLoader:
#   chooses which indexes to load
#   groups samples into batches
#   optionally shuffles indexes
#   optionally loads in parallel
#   stacks tensors together

# this is just an an exercise, i wont be using this.
import random
from collections.abc import Iterator

from torch import Tensor, stack, tensor

from trail_lens.dataset import TrailLensDataset


class TrailLensDataLoader:
    def __init__(self, dataset: TrailLensDataset, batch_size: int, shuffle: bool) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        indexes = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indexes)

        for start in range(0, len(indexes), self.batch_size):
            batch_indexes = indexes[start : start + self.batch_size]

            images = []
            labels = []

            for index in batch_indexes:
                image, label = self.dataset[index]
                images.append(image)
                labels.append(label)

            batch_images = stack(images)
            batch_labels = tensor(labels)

            yield batch_images, batch_labels
