# Run 1
This is my very first training loop. 
Dataset is not really cleaned, there is a lot bad data still (probably).
Using the sample code from the pytorch tutorial.
Adding image resize `v2.Resize(size=(224, 224), antialias=True)`.

```bash
Shape of X [N, C, H, W]: torch.Size([64, 3, 224, 224])
Shape of y: torch.Size([64]) torch.int64
Shape of X [N, C, H, W]: torch.Size([64, 3, 224, 224])
Shape of y: torch.Size([64]) torch.int64
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=150528, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
Epoch 1
-------------------------------
loss: 2.303005  [   64/  816]
Test Error: 
 Accuracy: 25.2%, Avg loss: 2.279041 
Epoch 2
-------------------------------
loss: 2.277936  [   64/  816]
Test Error: 
 Accuracy: 24.9%, Avg loss: 2.267105 
Epoch 3
-------------------------------
loss: 2.242252  [   64/  816]
Test Error: 
 Accuracy: 25.2%, Avg loss: 2.250837 
Epoch 4
-------------------------------
loss: 2.256196  [   64/  816]
Test Error: 
 Accuracy: 23.4%, Avg loss: 2.239223 
Epoch 5
-------------------------------
loss: 2.238975  [   64/  816]
Test Error: 
 Accuracy: 24.5%, Avg loss: 2.221466 

```

First step flatten the model from `[3,224,224]` into `150528` numbers. 
This means the model see the image as a big list of pixels.
It doesn't understand:
- edges
- shapes
- spatial relationships ... 
A CNNs model is better for this, as it can learn edges, curves, shapes and so ...

Second layer is massive `(0): Linear(in_features=150528, out_features=512, bias=True)` for such a small dataset.


In my first model run you can see that the loss is `2.279041 ` this is expected, as the model is randomly guessing.

As the iteration go on the loss decreases, `2.221466`.

This means the model became more confident on the correct answer.




# Loss
Loss measures how wrong/confident the model is. Lower is better.

For an uniformly random classifier under cross-entropy loss:

$$
L = -\ln\left(\frac{1}{C}\right) = \ln(C)
$$

Where:

L = expected loss
C = number of classes
 
**Batch loss** moment-to-moment behavior, can get a bit noisy and fluctuate.
 
**Average epoch loss** It is the more important metric for observing training progress. It should decrease over epoch. indicates that the optimizer is improving the model, and that the model is learning patterns.

# Accuracy
Accuracy is basically, `correct predictions / total predictions`

How to compare if it is learning? 
    Initially compare against the random guess `1 / number of classes`

# On loss and Accuracy

Prediction A
| Class      | Probability |
| ---------- | ----------- |
| oak        | 51%         |
| poison ivy | 49%         |
Prediction B
| Class      | Probability |
| ---------- | ----------- |
| oak        | 99%         |
| poison ivy | 1%          |

Both predictions are, wrong same accuracy outcome. Prediction A is **MUCH** better. In prediction B the model is almost certain of the wrong thing!

Loss captures this difference.

Accuracy does not.

loss is smoother

accuracy is more jumpy/coarse

# ReLU()
ReLU is an activation function. It keeps positive numbers and turns negative numbers into zero.
Conceptually:
ReLU(-3) = 0
ReLU(5) = 5


I should keep track of train loss as well.

| Metric                  | Why                                      |
| ----------------------- | ---------------------------------------- |
| batch train loss        | live/noisy training signal               |
| average train loss      | is the model fitting training data?      |
| train accuracy          | how often it gets training samples right |
| average validation loss | is it generalizing?                      |
| validation accuracy     | how often it gets unseen samples right   |

```bash
overfitting 
Train acc: 85%
Val acc: 30%
```

```bash
underfitting
Train acc: 65%
Val acc: 60%
```


# Run 2


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
loss: 2.286090  [   64/  816]
Train Error: 
 Accuracy: 12.4%, Avg loss: 2.315696 

Test Error: 
 Accuracy: 15.3%, Avg loss: 2.274731 

Epoch 2
-------------------------------
loss: 2.258419  [   64/  816]
Train Error: 
 Accuracy: 17.9%, Avg loss: 2.231441 

Test Error: 
 Accuracy: 20.4%, Avg loss: 2.150279 

Epoch 3
-------------------------------
loss: 2.119792  [   64/  816]
Train Error: 
 Accuracy: 26.7%, Avg loss: 2.048683 

Test Error: 
 Accuracy: 26.7%, Avg loss: 1.997069 

Epoch 4
-------------------------------
loss: 1.979973  [   64/  816]
Train Error: 
 Accuracy: 33.7%, Avg loss: 1.855946 

Test Error: 
 Accuracy: 29.7%, Avg loss: 1.946249 

Epoch 5
-------------------------------
loss: 1.584085  [   64/  816]
Train Error: 
 Accuracy: 40.7%, Avg loss: 1.688038 

Test Error: 
 Accuracy: 35.2%, Avg loss: 1.855946 

Done!

```

| Metric            | Epoch 1 | Epoch 5 | Direction |
| ----------------- | ------: | ------: | --------- |
| Train accuracy    |   12.4% |   40.7% | up        |
| Train loss        |   2.315 |   1.688 | down      |
| Val/Test accuracy |   15.3% |   35.2% | up        |
| Val/Test loss     |   2.274 |   1.855 | down      |


resize transform to 96x96

```bash
Using cuda device
Shape of X [N, C, H, W]: torch.Size([64, 3, 96, 96])
Shape of y: torch.Size([64]) torch.int64
Shape of X [N, C, H, W]: torch.Size([64, 3, 96, 96])
Shape of y: torch.Size([64]) torch.int64
NeuralNetwork(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=7056, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
Epoch 1
-------------------------------
loss: 2.297900  [   64/  816]
Train Error: 
 Accuracy: 12.0%, Avg loss: 2.293057 

Test Error: 
 Accuracy: 14.2%, Avg loss: 2.260908 

Epoch 2
-------------------------------
loss: 2.238877  [   64/  816]
Train Error: 
 Accuracy: 19.0%, Avg loss: 2.186221 

Test Error: 
 Accuracy: 21.5%, Avg loss: 2.163307 

Epoch 3
-------------------------------
loss: 2.066007  [   64/  816]
Train Error: 
 Accuracy: 22.8%, Avg loss: 2.077438 

Test Error: 
 Accuracy: 23.6%, Avg loss: 2.069901 

Epoch 4
-------------------------------
loss: 1.864749  [   64/  816]
Train Error: 
 Accuracy: 29.7%, Avg loss: 1.969575 

Test Error: 
 Accuracy: 26.5%, Avg loss: 2.000427 

Epoch 5
-------------------------------
loss: 1.761043  [   64/  816]
Train Error: 
 Accuracy: 31.6%, Avg loss: 1.898482 

Test Error: 
 Accuracy: 28.2%, Avg loss: 1.928335 

Done!
```

resize transform to 120x120

```bash

Using cuda device
Shape of X [N, C, H, W]: torch.Size([64, 3, 128, 128])
Shape of y: torch.Size([64]) torch.int64
Shape of X [N, C, H, W]: torch.Size([64, 3, 128, 128])
Shape of y: torch.Size([64]) torch.int64
NeuralNetwork(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=13456, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
Epoch 1
-------------------------------
loss: 2.309970  [   64/  816]
Train Error: 
 Accuracy: 13.1%, Avg loss: 2.290506 

Test Error: 
 Accuracy: 15.5%, Avg loss: 2.261800 

Epoch 2
-------------------------------
loss: 2.274948  [   64/  816]
Train Error: 
 Accuracy: 19.4%, Avg loss: 2.166450 

Test Error: 
 Accuracy: 24.5%, Avg loss: 2.100682 

Epoch 3
-------------------------------
loss: 2.019582  [   64/  816]
Train Error: 
 Accuracy: 26.6%, Avg loss: 2.039491 

Test Error: 
 Accuracy: 26.9%, Avg loss: 2.031868 

Epoch 4
-------------------------------
loss: 2.022979  [   64/  816]
Train Error: 
 Accuracy: 28.3%, Avg loss: 1.975737 

Test Error: 
 Accuracy: 27.1%, Avg loss: 2.033791 

Epoch 5
-------------------------------
loss: 1.949794  [   64/  816]
Train Error: 
 Accuracy: 29.3%, Avg loss: 1.917434 

Test Error: 
 Accuracy: 27.6%, Avg loss: 2.014519 

Done!

```


| Input Size | Train Acc (Epoch 5) | Val/Test Acc (Epoch 5) | Val Loss |
| ---------- | ------------------: | ---------------------: | -------: |
| `96x96`    |               31.6% |                  28.2% |    1.928 |
| `128x128`  |               29.3% |                  27.6% |    2.015 |
| `224x224`  |               40.7% |                  35.2% |    1.855 |


## Mind the gap

The gap between:

How good is the model at memorizing training data
(train accuracy)

vs

How good is it at handling new unseen data
(validation/test accuracy)

The gap between fitting ability (train accuracy)
and generalization ability (validation/test accuracy).


| State | Train Accuracy | Validation/Test Accuracy | Gap | Interpretation |
|---|---|---|---|---|
| Underfitting | Low | Low | Small | Model cannot learn the training data well |
| Healthy fitting | High | High | Small / Moderate | Model generalizes well to unseen data |
| Overfitting | Very High | Lower | Large | Model memorizes training-specific information |


## Explore
- add more epochs to training
- add augmentation
  - random horizontal flip
  - random rotation
  - random resized crop
  - slight color jitter
- transfer learning

  This means taking an already pre-trained model ona huge dataset and adapt it to our task.
  Some model could be:
    - MobileNetV3
    - ResNet18
    - EfficientNet
    - ConvNeXt

    

  Can be done with **feature extraction** or **fine tuning**
- normalization

  Neural networks train better when inputs are centered and scaled consistently.
  This allows better optimization, the optimizer doesn't need to fight against a constant bias and inconsistent scale present that is present in input distribution.

  ````
  No normalization
  the steering wheel constantly pulls right -> So every correction must compensate for that bias.

  With normalization
  wheel starts centered -> corrections are smoother
  ````

  When using a pre-trained model, ALWAYS use the expected normalization 
- learning rate
  
  If loss jumps around too much, LR is probably too high.
  If loss barely moves, LR may be too low.
- batch size

  Smaller batch can sometimes generalize better. Larger batch can train faster but may need LR changes.
- optimizer experiments

  This tells the model how to update weights using the computed gradients. How big updates are.

  `loss -> gradients -> optimizer updates weights`

  Common optimizers:
  - SGD
  - SGD + momentum
  - Adam
  - AdamW
  - RMSProp

- loss function experiments

  This measures how wrong the predictions are compared to the true labels.
  
  Some losses also penalize overconfident wrong predictions more strongly. With `CrossEntropyLoss` the punishment grows rapidly:
$$
L = -\log(p_{true})
$$
$$
L = - \sum_{i=1}^{C} y_i \log(p_i)
$$

  Examples:
  - CrossEntropyLoss
  - BCEWithLogitsLoss
  - Focal loss
  - Label smoothing variants
