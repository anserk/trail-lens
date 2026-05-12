from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from trail_lens.dataset import TrailLensDataset
from trail_lens.split import TrailLensDataSubset, train_val_split

# note this is still pretty much copy and paste from pytorch tutorial.

accelerator = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
device = accelerator.type if accelerator is not None else "cpu"
print(f"Using {device} device")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
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


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
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
    test_dataloader = DataLoader(
        TrailLensDataSubset(dataset, validation_indexes, validation_transform),
        batch_size=batch_size,
        shuffle=False,
    )

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Define model
    import torch.nn as nn
    import torch.nn.functional as F

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 53 * 53, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

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
        current_accuracy, current_loss = test(test_dataloader, model, loss_fn)
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
