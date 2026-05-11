import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image

from trail_lens.classes import CLASS_TO_IDX


class TrailLensDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        transform: Callable[[Tensor], Tensor] | None = None,
        target_transform: Callable[[int], int] | None = None,
    ) -> None:
        self.image_labels: list[dict[str, Any]] = []
        self.class_to_idx = CLASS_TO_IDX
        index = 0

        for entry in os.listdir(image_dir):
            current_path = image_dir / entry
            if os.path.isdir(current_path):
                label_id = self.class_to_idx[entry]
                metadata_path = current_path / "metadata.jsonl"
                with open(metadata_path) as f:
                    for line in f:
                        line_data = json.loads(line)
                        line_data["index"] = index
                        line_data["label"] = entry
                        line_data["label_id"] = label_id
                        line_data["full_image_path"] = str(current_path / line_data["file"])
                        self.image_labels.append(line_data)
                        index += 1

        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.image_labels)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_metadata = self.image_labels[index]
        image = decode_image(image_metadata["full_image_path"], ImageReadMode.RGB)
        label = int(image_metadata["label_id"])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = TrailLensDataset(Path("data/raw_candidates"))
    print(f"Dataset length: {len(dataset)}")
    image, label = dataset[11]
    print(f"Image shape: {tuple(image.shape)}")
    print(f"Image dtype: {image.dtype}")
    print(f"Label: {label}")

    plt.imshow(image.permute(1, 2, 0))
    plt.title(f"Label: {label}")
    plt.axis("off")
    if plt.get_backend().lower().endswith("agg"):
        preview_path = Path("dataset_preview.png")
        plt.savefig(preview_path, bbox_inches="tight")
        print(f"Saved preview to {preview_path}")
    else:
        plt.show()
