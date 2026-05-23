import logging

import torch

from trail_lens.model import NeuralNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch_model = NeuralNetwork()
torch_model.load_state_dict(torch.load("models/20260511_084105/model_weights_11_best_accuracy.pth"))
torch_model.eval()
# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = (torch.randn(1, 3, 224, 224),)
onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
if onnx_program is not None:
    onnx_program.save("image_classifier_model.onnx")
else:
    logger.error("Unable to save model.")
