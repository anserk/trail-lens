# Inference 

For inference, we need to use the same model architecture and load the saved weights. We can either load best accuracy or best loss.


```python
image = transform(image)
# [3, 224, 224] 
# Model expects a tensor with the same preprocessing used during training.

image = image.unsqueeze(0)
# [1, 3, 224, 224] 
# Add batch dimension, model expect batches.

# run model without tracking gradients.
with torch.inference_mode():
    prediction = model(image)
```

```
probabilities = torch.softmax(prediction, dim=1)
```

A model outputs logits, so we use softmax to convert them into probabilites. 

The output could be something similar to
```python
tensor([[2.1, 0.3, -1.2]])
# shape[1, 3] 
# [batch_size, num_classes]
```

Given the following classes `["red maple", "white oak", "poison ivy"]`

```
logits:
red maple = 2.1
white oak = 0.3
poison ivy = -1.2

↓ softmax

probabilities:
red maple = 0.82
white oak = 0.15
poison ivy = 0.03
```

We use `dim=1` on softmax beacuse that's the dimension we care about (classes).

```
confidence, predicted_index = probabilities.max(dim=1)
```
This returns the highest probability and index of that probability.
```
confidence = tensor([0.82])
predicted_index = tensor([0])
```
