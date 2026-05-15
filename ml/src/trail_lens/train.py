from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from trail_lens.classes import IDX_TO_CLASS
from trail_lens.dataset import TrailLensDataset
from trail_lens.model import NeuralNetwork
from trail_lens.split import TrailLensDataSubset, train_val_split

# note this is still pretty much copy and paste from pytorch tutorial.

accelerator = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
device = accelerator.type if accelerator is not None else "cpu"
print(f"Using {device} device")


def train(
    dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer
) -> None:
    size = len(dataloader.dataset)  # ty:ignore[invalid-argument-type]
    num_batches = len(dataloader)

    model.train()

    train_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size

    print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")


def validate(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module) -> tuple[float, float]:
    size = len(dataloader.dataset)  # ty:ignore[invalid-argument-type]
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def main() -> None:
    dataset = TrailLensDataset(
        image_dir=Path("data/raw_candidates"),
    )

    validation_transform = v2.Compose(
        [
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    training_transform = v2.Compose(
        [
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),  # ty:ignore[invalid-argument-type]
            # v2.ColorJitter(
            #     brightness=0.1,
            #     contrast=0.1,
            #     saturation=0.1,
            #     hue=0.02,
            # ),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    training_indexes, validation_indexes = train_val_split(dataset, 0.4, 42)

    batch_size = 64

    train_dataloader = DataLoader(
        TrailLensDataSubset(dataset, training_indexes, training_transform),
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        TrailLensDataSubset(dataset, validation_indexes, validation_transform),
        batch_size=batch_size,
        shuffle=False,
    )

    for X, y in val_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        print(f"Len of val_dataloader: {len(val_dataloader.dataset)}")  # ty:ignore[invalid-argument-type]
        print(val_dataloader.dataset[0][1])
        break

    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        print(f"Len of train_dataloader: {len(train_dataloader.dataset)}")  # ty:ignore[invalid-argument-type]
        break

    # from collections import Counter

    # def print_class_distribution(name: str, counts: dict[int, int], total: int) -> None:
    #     print(f"--- {name} ---")
    #     for cls_id in sorted(counts):
    #         count = counts[cls_id]
    #         print(IDX_TO_CLASS[cls_id], count, f"{(count * 100 / total):>0.1f}%")

    # val_class_count: Counter[Any] = Counter(
    #     dataset.image_labels[i]["label_id"] for i in validation_indexes
    # )
    # train_class_count: Counter[Any] = Counter(
    #     dataset.image_labels[i]["label_id"] for i in training_indexes
    # )

    # print_class_distribution("validation", val_class_count, len(validation_indexes))
    # print_class_distribution("train", train_class_count, len(training_indexes))

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    best_accuracy = 0
    best_model_path: Path | None = None

    best_loss = float("inf")
    best_loss_path: Path | None = None

    model_path = Path("models") / datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path.mkdir(parents=True, exist_ok=True)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        current_accuracy, current_loss = validate(val_dataloader, model, loss_fn)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            current_model_path = model_path / f"model_weights_{t + 1}_best_accuracy.pth"
            torch.save(model.state_dict(), current_model_path)
            if best_model_path is not None:
                best_model_path.unlink()
            best_model_path = current_model_path
        if current_loss < best_loss:
            best_loss = current_loss
            current_model_path = model_path / f"model_weights_{t + 1}_best_loss.pth"
            torch.save(model.state_dict(), current_model_path)
            if best_loss_path is not None:
                best_loss_path.unlink()
            best_loss_path = current_model_path

    print("Done!")


if __name__ == "__main__":
    main()
