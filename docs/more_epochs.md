10 epoch at 224x224

```bash
Using cuda device
Shape of X [N, C, H, W]: torch.Size([64, 3, 224, 224])
Shape of y: torch.Size([64]) torch.int64
Shape of X [N, C, H, W]: torch.Size([64, 3, 224, 224])
Shape of y: torch.Size([64]) torch.int64
NeuralNetwork(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=44944, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
Epoch 1
-------------------------------
loss: 2.312636  [   64/  816]
Train Error: 
 Accuracy: 14.8%, Avg loss: 2.218413 

Test Error: 
 Accuracy: 21.7%, Avg loss: 2.120124 

Epoch 2
-------------------------------
loss: 2.023698  [   64/  816]
Train Error: 
 Accuracy: 23.3%, Avg loss: 2.091498 

Test Error: 
 Accuracy: 21.7%, Avg loss: 2.116842 

Epoch 3
-------------------------------
loss: 2.084882  [   64/  816]
Train Error: 
 Accuracy: 25.1%, Avg loss: 2.019960 

Test Error: 
 Accuracy: 26.9%, Avg loss: 2.056951 

Epoch 4
-------------------------------
loss: 1.968322  [   64/  816]
Train Error: 
 Accuracy: 31.2%, Avg loss: 1.909637 

Test Error: 
 Accuracy: 30.2%, Avg loss: 2.049589 

Epoch 5
-------------------------------
loss: 1.868603  [   64/  816]
Train Error: 
 Accuracy: 36.3%, Avg loss: 1.814357 

Test Error: 
 Accuracy: 31.7%, Avg loss: 1.965415 

Epoch 6
-------------------------------
loss: 1.754887  [   64/  816]
Train Error: 
 Accuracy: 43.0%, Avg loss: 1.672299 

Test Error: 
 Accuracy: 31.7%, Avg loss: 1.970510 

Epoch 7
-------------------------------
loss: 1.449815  [   64/  816]
Train Error: 
 Accuracy: 50.2%, Avg loss: 1.467542 

Test Error: 
 Accuracy: 34.6%, Avg loss: 1.862790 

Epoch 8
-------------------------------
loss: 1.237191  [   64/  816]
Train Error: 
 Accuracy: 59.9%, Avg loss: 1.229757 

Test Error: 
 Accuracy: 35.0%, Avg loss: 2.015193 

Epoch 9
-------------------------------
loss: 0.979921  [   64/  816]
Train Error: 
 Accuracy: 65.6%, Avg loss: 1.034540 

Test Error: 
 Accuracy: 41.1%, Avg loss: 1.919609 

Epoch 10
-------------------------------
loss: 0.870855  [   64/  816]
Train Error: 
 Accuracy: 78.6%, Avg loss: 0.748874 

Test Error: 
 Accuracy: 39.2%, Avg loss: 2.237949 

Done!
```

There is a growing gap between training accuracy and validation/test accuracy, meaning the model is learning the training data better than it generalizes to unseen data.

Model is overfitting, you can notice this around epoch 5/6.

## Next

**add augmentation**