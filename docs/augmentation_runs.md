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
loss: 2.299350  [   64/  816]
Train Error:
 Accuracy: 16.8%, Avg loss: 2.248144

Test Error:
 Accuracy: 23.4%, Avg loss: 2.129613

Epoch 2
-------------------------------
loss: 2.072320  [   64/  816]
Train Error:
 Accuracy: 24.4%, Avg loss: 2.075071

Test Error:
 Accuracy: 30.2%, Avg loss: 2.017894

Epoch 3
-------------------------------
loss: 1.973996  [   64/  816]
Train Error:
 Accuracy: 29.4%, Avg loss: 1.920261

Test Error:
 Accuracy: 29.5%, Avg loss: 1.951412

Epoch 4
-------------------------------
loss: 1.726634  [   64/  816]
Train Error:
 Accuracy: 38.7%, Avg loss: 1.799868

Test Error:
 Accuracy: 33.5%, Avg loss: 1.934105

Epoch 5
-------------------------------
loss: 1.750148  [   64/  816]
Train Error:
 Accuracy: 42.3%, Avg loss: 1.695802

Test Error:
 Accuracy: 34.8%, Avg loss: 1.838172

Epoch 6
-------------------------------
loss: 1.508512  [   64/  816]
Train Error:
 Accuracy: 48.3%, Avg loss: 1.530826

Test Error:
 Accuracy: 39.2%, Avg loss: 1.765986

Epoch 7
-------------------------------
loss: 1.445001  [   64/  816]
Train Error:
 Accuracy: 54.8%, Avg loss: 1.385054

Test Error:
 Accuracy: 42.2%, Avg loss: 1.756020

Epoch 8
-------------------------------
loss: 1.307331  [   64/  816]
Train Error:
 Accuracy: 61.0%, Avg loss: 1.232288

Test Error:
 Accuracy: 41.8%, Avg loss: 1.796968

Epoch 9
-------------------------------
loss: 1.200023  [   64/  816]
Train Error:
 Accuracy: 62.9%, Avg loss: 1.133860

Test Error:
 Accuracy: 39.2%, Avg loss: 1.812968

Epoch 10
-------------------------------
loss: 1.079998  [   64/  816]
Train Error:
 Accuracy: 65.3%, Avg loss: 1.066133

Test Error:
 Accuracy: 39.0%, Avg loss: 1.875611

Done!
```


Adding horizontal flip reduced overfitting.

Compared to the previous 224x224 run, the final training accuracy was lower and the gap was smaller, meaning augmentation helped reduce memorization.

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
loss: 2.288051  [   64/  816]
Train Error:
 Accuracy: 13.5%, Avg loss: 2.395411

Test Error:
 Accuracy: 13.8%, Avg loss: 2.242769

Epoch 2
-------------------------------
loss: 2.263126  [   64/  816]
Train Error:
 Accuracy: 20.3%, Avg loss: 2.186384

Test Error:
 Accuracy: 21.9%, Avg loss: 2.144844

Epoch 3
-------------------------------
loss: 1.989183  [   64/  816]
Train Error:
 Accuracy: 21.9%, Avg loss: 2.104354

Test Error:
 Accuracy: 23.4%, Avg loss: 2.130206

Epoch 4
-------------------------------
loss: 2.064140  [   64/  816]
Train Error:
 Accuracy: 24.6%, Avg loss: 2.078952

Test Error:
 Accuracy: 16.9%, Avg loss: 2.137959

Epoch 5
-------------------------------
loss: 2.119529  [   64/  816]
Train Error:
 Accuracy: 22.1%, Avg loss: 2.060904

Test Error:
 Accuracy: 25.6%, Avg loss: 2.094981

Epoch 6
-------------------------------
loss: 2.107916  [   64/  816]
Train Error:
 Accuracy: 26.7%, Avg loss: 2.026889

Test Error:
 Accuracy: 26.5%, Avg loss: 2.088768

Epoch 7
-------------------------------
loss: 2.013935  [   64/  816]
Train Error:
 Accuracy: 25.1%, Avg loss: 2.014658

Test Error:
 Accuracy: 25.4%, Avg loss: 2.062776

Epoch 8
-------------------------------
loss: 1.927158  [   64/  816]
Train Error:
 Accuracy: 28.1%, Avg loss: 1.961076

Test Error:
 Accuracy: 25.8%, Avg loss: 1.995703

Epoch 9
-------------------------------
loss: 1.940788  [   64/  816]
Train Error:
 Accuracy: 30.6%, Avg loss: 1.896878

Test Error:
 Accuracy: 28.9%, Avg loss: 1.929492

Epoch 10
-------------------------------
loss: 1.915326  [   64/  816]
Train Error:
 Accuracy: 37.5%, Avg loss: 1.770243

Test Error:
 Accuracy: 28.9%, Avg loss: 2.032290

Done!
```

Adding `v2.RandomRotation(degrees=10)` slowed down the learning rate but the model fit looks healthy. 

Add more epochs and see how the learning goes.

for now best is still is

```bash
224x224 + horizontal flip
Best test: 42.2% at epoch 7
```

20 epochs with rotation and flip

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
loss: 2.310734  [   64/  816]
Train Error:
 Accuracy: 15.6%, Avg loss: 2.259935

Test Error:
 Accuracy: 16.6%, Avg loss: 2.181303

Epoch 2
-------------------------------
loss: 2.075953  [   64/  816]
Train Error:
 Accuracy: 23.4%, Avg loss: 2.125828

Test Error:
 Accuracy: 24.5%, Avg loss: 2.098946

Epoch 3
-------------------------------
loss: 2.035853  [   64/  816]
Train Error:
 Accuracy: 24.1%, Avg loss: 2.055708

Test Error:
 Accuracy: 25.0%, Avg loss: 2.041944

Epoch 4
-------------------------------
loss: 1.895758  [   64/  816]
Train Error:
 Accuracy: 29.5%, Avg loss: 1.972418

Test Error:
 Accuracy: 24.7%, Avg loss: 2.027472

Epoch 5
-------------------------------
loss: 2.062778  [   64/  816]
Train Error:
 Accuracy: 32.8%, Avg loss: 1.889909

Test Error:
 Accuracy: 33.9%, Avg loss: 1.898187

Epoch 6
-------------------------------
loss: 1.711129  [   64/  816]
Train Error:
 Accuracy: 36.3%, Avg loss: 1.778059

Test Error:
 Accuracy: 36.6%, Avg loss: 1.836612

Epoch 7
-------------------------------
loss: 1.527689  [   64/  816]
Train Error:
 Accuracy: 39.3%, Avg loss: 1.704231

Test Error:
 Accuracy: 35.0%, Avg loss: 1.914764

Epoch 8
-------------------------------
loss: 1.644203  [   64/  816]
Train Error:
 Accuracy: 38.6%, Avg loss: 1.739244

Test Error:
 Accuracy: 35.5%, Avg loss: 1.805253

Epoch 9
-------------------------------
loss: 1.566730  [   64/  816]
Train Error:
 Accuracy: 45.0%, Avg loss: 1.586344

Test Error:
 Accuracy: 38.1%, Avg loss: 1.798083

Epoch 10
-------------------------------
loss: 1.559386  [   64/  816]
Train Error:
 Accuracy: 49.8%, Avg loss: 1.472748

Test Error:
 Accuracy: 37.8%, Avg loss: 1.839620

Epoch 11
-------------------------------
loss: 1.593452  [   64/  816]
Train Error:
 Accuracy: 50.9%, Avg loss: 1.399522

Test Error:
 Accuracy: 38.9%, Avg loss: 1.758647

Epoch 12
-------------------------------
loss: 1.310834  [   64/  816]
Train Error:
 Accuracy: 54.4%, Avg loss: 1.336031

Test Error:
 Accuracy: 37.4%, Avg loss: 1.829310

Epoch 13
-------------------------------
loss: 0.990230  [   64/  816]
Train Error:
 Accuracy: 54.3%, Avg loss: 1.295209

Test Error:
 Accuracy: 42.0%, Avg loss: 1.782546

Epoch 14
-------------------------------
loss: 1.169840  [   64/  816]
Train Error:
 Accuracy: 59.4%, Avg loss: 1.232673

Test Error:
 Accuracy: 36.6%, Avg loss: 1.870359

Epoch 15
-------------------------------
loss: 1.174291  [   64/  816]
Train Error:
 Accuracy: 62.1%, Avg loss: 1.133306

Test Error:
 Accuracy: 41.1%, Avg loss: 1.816637

Epoch 16
-------------------------------
loss: 0.982959  [   64/  816]
Train Error:
 Accuracy: 60.8%, Avg loss: 1.113190

Test Error:
 Accuracy: 39.6%, Avg loss: 1.871088

Epoch 17
-------------------------------
loss: 1.181797  [   64/  816]
Train Error:
 Accuracy: 63.2%, Avg loss: 1.063614

Test Error:
 Accuracy: 39.2%, Avg loss: 1.879604

Epoch 18
-------------------------------
loss: 0.967530  [   64/  816]
Train Error:
 Accuracy: 66.8%, Avg loss: 0.942321

Test Error:
 Accuracy: 39.8%, Avg loss: 1.887415

Epoch 19
-------------------------------
loss: 0.919852  [   64/  816]
Train Error:
 Accuracy: 71.4%, Avg loss: 0.864052

Test Error:
 Accuracy: 40.1%, Avg loss: 2.038860

Epoch 20
-------------------------------
loss: 1.087501  [   64/  816]
Train Error:
 Accuracy: 68.4%, Avg loss: 0.937111

Test Error:
 Accuracy: 42.7%, Avg loss: 1.892422

Done!
```

New best validation score
```bash
224x224 + horizontal flip + rotation
Best test: 42.7% at epoch 20
```
However, later epochs show overfitting again: training accuracy continued increasing while validation/test accuracy stayed mostly flat.

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
loss: 2.288283  [   64/  816]
Train Error:
 Accuracy: 16.7%, Avg loss: 2.316142

Test Error:
 Accuracy: 22.8%, Avg loss: 2.126293

Epoch 2
-------------------------------
loss: 2.108779  [   64/  816]
Train Error:
 Accuracy: 23.4%, Avg loss: 2.062264

Test Error:
 Accuracy: 26.3%, Avg loss: 2.083738

Epoch 3
-------------------------------
loss: 1.867245  [   64/  816]
Train Error:
 Accuracy: 27.2%, Avg loss: 1.990015

Test Error:
 Accuracy: 28.4%, Avg loss: 2.016906

Epoch 4
-------------------------------
loss: 2.081473  [   64/  816]
Train Error:
 Accuracy: 33.9%, Avg loss: 1.902606

Test Error:
 Accuracy: 31.9%, Avg loss: 1.996273

Epoch 5
-------------------------------
loss: 1.536856  [   64/  816]
Train Error:
 Accuracy: 36.2%, Avg loss: 1.827390

Test Error:
 Accuracy: 34.8%, Avg loss: 1.906732

Epoch 6
-------------------------------
loss: 1.941076  [   64/  816]
Train Error:
 Accuracy: 43.1%, Avg loss: 1.692794

Test Error:
 Accuracy: 36.5%, Avg loss: 1.888322

Epoch 7
-------------------------------
loss: 1.561542  [   64/  816]
Train Error:
 Accuracy: 43.5%, Avg loss: 1.670907

Test Error:
 Accuracy: 36.8%, Avg loss: 1.875102

Epoch 8
-------------------------------
loss: 1.483092  [   64/  816]
Train Error:
 Accuracy: 45.1%, Avg loss: 1.579174

Test Error:
 Accuracy: 36.1%, Avg loss: 1.869019

Epoch 9
-------------------------------
loss: 1.493542  [   64/  816]
Train Error:
 Accuracy: 46.9%, Avg loss: 1.518864

Test Error:
 Accuracy: 37.6%, Avg loss: 1.835701

Epoch 10
-------------------------------
loss: 1.354580  [   64/  816]
Train Error:
 Accuracy: 50.5%, Avg loss: 1.462949

Test Error:
 Accuracy: 38.5%, Avg loss: 1.781309

Epoch 11
-------------------------------
loss: 1.175209  [   64/  816]
Train Error:
 Accuracy: 54.0%, Avg loss: 1.339985

Test Error:
 Accuracy: 42.5%, Avg loss: 1.797651

Epoch 12
-------------------------------
loss: 1.050677  [   64/  816]
Train Error:
 Accuracy: 54.8%, Avg loss: 1.313691

Test Error:
 Accuracy: 39.4%, Avg loss: 1.881165

Epoch 13
-------------------------------
loss: 1.043350  [   64/  816]
Train Error:
 Accuracy: 59.9%, Avg loss: 1.194373

Test Error:
 Accuracy: 43.5%, Avg loss: 1.806356

Epoch 14
-------------------------------
loss: 1.323880  [   64/  816]
Train Error:
 Accuracy: 62.9%, Avg loss: 1.079765

Test Error:
 Accuracy: 42.5%, Avg loss: 1.894011

Epoch 15
-------------------------------
loss: 1.014257  [   64/  816]
Train Error:
 Accuracy: 65.3%, Avg loss: 1.037782

Test Error:
 Accuracy: 41.1%, Avg loss: 1.936789

Epoch 16
-------------------------------
loss: 1.108295  [   64/  816]
Train Error:
 Accuracy: 64.6%, Avg loss: 1.015388

Test Error:
 Accuracy: 42.5%, Avg loss: 1.880522

Epoch 17
-------------------------------
loss: 1.097776  [   64/  816]
Train Error:
 Accuracy: 67.5%, Avg loss: 0.943680

Test Error:
 Accuracy: 42.4%, Avg loss: 2.002483

Epoch 18
-------------------------------
loss: 0.726475  [   64/  816]
Train Error:
 Accuracy: 71.9%, Avg loss: 0.814510

Test Error:
 Accuracy: 39.4%, Avg loss: 2.381627

Epoch 19
-------------------------------
loss: 0.726935  [   64/  816]
Train Error:
 Accuracy: 75.4%, Avg loss: 0.763529

Test Error:
 Accuracy: 45.7%, Avg loss: 2.258939

Epoch 20
-------------------------------
loss: 0.664160  [   64/  816]
Train Error:
 Accuracy: 77.3%, Avg loss: 0.689584

Test Error:
 Accuracy: 45.3%, Avg loss: 2.138373

Done!
```

New best validation score
```bash
224x224 + horizontal flip + rotation
Best test: 45.7% at epoch 19
```
However, the model is still overfitting. Training accuracy keeps increasing, while validation/test accuracy improves only slightly and validation/test loss rises.

## Augmentation should be applied only to training dataset
As the title say, augmentation should be applied only to training dataset. The validation dataset should be stable, deterministic, non-random.

I am going to fix this issue and re-try a run. 

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
loss: 2.307547  [   64/  816]
Train Error:
 Accuracy: 16.5%, Avg loss: 2.260558

Test Error:
 Accuracy: 21.7%, Avg loss: 2.085725

Epoch 2
-------------------------------
loss: 2.078984  [   64/  816]
Train Error:
 Accuracy: 26.2%, Avg loss: 2.025369

Test Error:
 Accuracy: 32.0%, Avg loss: 1.994210

Epoch 3
-------------------------------
loss: 2.068070  [   64/  816]
Train Error:
 Accuracy: 34.2%, Avg loss: 1.890619

Test Error:
 Accuracy: 29.5%, Avg loss: 2.007565

Epoch 4
-------------------------------
loss: 1.849519  [   64/  816]
Train Error:
 Accuracy: 35.9%, Avg loss: 1.812373

Test Error:
 Accuracy: 36.6%, Avg loss: 1.814795

Epoch 5
-------------------------------
loss: 1.610279  [   64/  816]
Train Error:
 Accuracy: 43.4%, Avg loss: 1.666407

Test Error:
 Accuracy: 42.2%, Avg loss: 1.746846

Epoch 6
-------------------------------
loss: 1.654773  [   64/  816]
Train Error:
 Accuracy: 46.2%, Avg loss: 1.564802

Test Error:
 Accuracy: 40.1%, Avg loss: 1.713659

Epoch 7
-------------------------------
loss: 1.419409  [   64/  816]
Train Error:
 Accuracy: 51.2%, Avg loss: 1.434347

Test Error:
 Accuracy: 40.0%, Avg loss: 1.766201

Epoch 8
-------------------------------
loss: 1.218372  [   64/  816]
Train Error:
 Accuracy: 54.2%, Avg loss: 1.349791

Test Error:
 Accuracy: 41.1%, Avg loss: 1.741719

Epoch 9
-------------------------------
loss: 1.259334  [   64/  816]
Train Error:
 Accuracy: 59.3%, Avg loss: 1.207465

Test Error:
 Accuracy: 43.8%, Avg loss: 1.714899

Epoch 10
-------------------------------
loss: 1.111925  [   64/  816]
Train Error:
 Accuracy: 64.2%, Avg loss: 1.048958

Test Error:
 Accuracy: 44.0%, Avg loss: 1.857356

Epoch 11
-------------------------------
loss: 1.039221  [   64/  816]
Train Error:
 Accuracy: 67.5%, Avg loss: 0.959281

Test Error:
 Accuracy: 45.9%, Avg loss: 1.819489

Epoch 12
-------------------------------
loss: 0.763980  [   64/  816]
Train Error:
 Accuracy: 70.7%, Avg loss: 0.902215

Test Error:
 Accuracy: 45.1%, Avg loss: 2.042067

Epoch 13
-------------------------------
loss: 0.666144  [   64/  816]
Train Error:
 Accuracy: 71.2%, Avg loss: 0.894953

Test Error:
 Accuracy: 43.8%, Avg loss: 2.064567

Epoch 14
-------------------------------
loss: 0.734559  [   64/  816]
Train Error:
 Accuracy: 75.0%, Avg loss: 0.782444

Test Error:
 Accuracy: 42.4%, Avg loss: 1.952943

Epoch 15
-------------------------------
loss: 0.528144  [   64/  816]
Train Error:
 Accuracy: 79.9%, Avg loss: 0.640489

Test Error:
 Accuracy: 43.6%, Avg loss: 1.994640

Epoch 16
-------------------------------
loss: 0.409943  [   64/  816]
Train Error:
 Accuracy: 84.8%, Avg loss: 0.489428

Test Error:
 Accuracy: 44.0%, Avg loss: 2.241579

Epoch 17
-------------------------------
loss: 0.533642  [   64/  816]
Train Error:
 Accuracy: 86.3%, Avg loss: 0.456840

Test Error:
 Accuracy: 42.7%, Avg loss: 2.306425

Epoch 18
-------------------------------
loss: 0.414647  [   64/  816]
Train Error:
 Accuracy: 85.3%, Avg loss: 0.430618

Test Error:
 Accuracy: 43.3%, Avg loss: 2.476370

Epoch 19
-------------------------------
loss: 0.615740  [   64/  816]
Train Error:
 Accuracy: 87.0%, Avg loss: 0.422452

Test Error:
 Accuracy: 41.8%, Avg loss: 2.545642

Epoch 20
-------------------------------
loss: 0.207358  [   64/  816]
Train Error:
 Accuracy: 91.5%, Avg loss: 0.323425

Test Error:
 Accuracy: 41.6%, Avg loss: 2.783201

Done!
```

New best validation score
```bash
224x224 + horizontal flip + rotation, augmentation only on training
Best test: 45.9% at epoch 11
```
However, the model is still overfitting.

## Next

- Save best checkpoint

    Every time there is a best validation accuracy results, save the model state to file for later inference usage.

- Early stopping

    Stop training when validation performance stops improving for several epochs. This should reduce overfitting.

    Look into adding `patience`, to quantify *several*.

- Add confusion matrix

    Generate a confusion matrix after best validation to better understand which classes the model confuses with one another.

    This can help identify:
    - visually similar classes
    - weak training data
    - class imbalance
    - labeling problems

- Inspect wrong prediction

    Save worst mistakes:
        - image path
        - true label
        - predicted label
        - confidence

    This should give me a better idea of what the model gets wrong. It could be an issue with the dataset: duplicate images, bad labels, bad validation split, bad image, wrong focus and so on ...

The main difference between confusion matrix and single wrong prediction is, confusion matrix focuses on patterns at the class level while inspect wrong prediction gives you details about a single failure.

Another way to put it:

| Technique | Main Question | Focus |
|---|---|---|
| Confusion matrix | "What kinds of mistakes happen repeatedly?" | Repeated class-level mistake patterns |
| Wrong prediction inspection | "Why did this specific prediction fail?" | Root cause analysis of individual failures |