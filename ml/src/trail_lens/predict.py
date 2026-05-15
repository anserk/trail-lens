import torch
from torch import Tensor
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

from trail_lens.classes import IDX_TO_CLASS
from trail_lens.model import NeuralNetwork

accelerator = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
device = accelerator.type if accelerator is not None else "cpu"
print(f"Using {device} device")


def predict() -> None:
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("models/20260511_084105/model_weights_11_best_accuracy.pth"))
    model.eval()

    image: Tensor = decode_image("src/trail_lens/oak.jpg", ImageReadMode.RGB)

    transform = v2.Compose(
        [
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.inference_mode():
        prediction = model(image)

    probabilities = torch.softmax(prediction, dim=1)

    for index, probability in enumerate(probabilities[0]):
        class_name = IDX_TO_CLASS[index]
        print(f"{class_name}: {probability.item():.2%}")

    confidence, predicted_index = probabilities.max(dim=1)
    predicted_index = predicted_index.item()
    confidence = confidence.item()
    print(IDX_TO_CLASS[int(predicted_index)])
    print(f"{confidence:.2%}")


if __name__ == "__main__":
    predict()
