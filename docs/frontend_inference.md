# Frontend Image Processing and Inference

## Base64

Base64 encodes data in chunks of 3 bytes (24 bits) into 4 base64 characters (6 bits each)

1 byte is 8 bits

Composed by the following alphabet

```python
const base64Alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
```

the encoded message can contain some padding characters

| 6 bit    | base64 | Byte |
| ------- | ------ | ---- |
| Padding | `=`    | 61   |

every 4 base64 characters represent up to 3 bytes.

### Encode

if the input is divisible by 3 then no padding
if the input is not divisible by 3

- missing 1 byte to complete a 3 byte group: add =
- missing 2 bytes to complete a 3 byte group: add ==

for this reason padding cannot be more than 2 characters.

### Decode

```python
  const outputLength =
    Math.floor((sanitized.length * 3) / 4) -
    (sanitized.endsWith("==") ? 2 : sanitized.endsWith("=") ? 1 : 0);

   const bytes = new Uint8Array(outputLength);


 [0,0,0...]
```

```python
  for (let i = 0; i < sanitized.length; i += 4) {
```

keep in mind every base64 block is 4 characters so we loop 4 at a time.

each base64 character gives 6 bits so 6 x 4 = 24 bit value

this converts the 4 characters into

```python
    const chunk =
      (base64Alphabet.indexOf(sanitized[i]) << 18) |
      (base64Alphabet.indexOf(sanitized[i + 1]) << 12) |
      ((base64Alphabet.indexOf(sanitized[i + 2]) & 63) << 6) |
      (base64Alphabet.indexOf(sanitized[i + 3]) & 63);
```

```bash
input = "Man"

base64 = "TWFu"

T -> 19 -> 010011
W -> 22 -> 010110
F -> 5  -> 000101
u -> 46 -> 101110

we should get to this output

010011 010110 000101 101110

01001101 01100001 01101110
M        a        n
77       97       110


base64Alphabet.indexOf(sanitized[i]) << 18
For T:
010011 << 18
010011 000000 000000 000000


base64Alphabet.indexOf(sanitized[i + 1]) << 12
For W:
010110 << 12
000000 010110 000000 000000

(base64Alphabet.indexOf(sanitized[i + 2]) & 63) << 6
For F:
000101 << 6
000000 000000 000101 000000

base64Alphabet.indexOf(sanitized[i + 3]) & 63
For u:
101110
000000 000000 000000 101110

those are all connected by | bitwise or

010011 000000 000000 000000
000000 010110 000000 000000
000000 000000 000101 000000
000000 000000 000000 101110
--------------------------------
010011 010110 000101 101110

chunk = 010011010110000101101110 (24 bit number)

writes the bytes

if (byteIndex < outputLength) bytes[byteIndex++] = (chunk >> 16) & 255;
    if (byteIndex < outputLength) bytes[byteIndex++] = (chunk >> 8) & 255;
    if (byteIndex < outputLength) bytes[byteIndex++] = chunk & 255;

(chunk >> 16) & 255
Keeps the first byte:
01001101 = 77 = "M"

(chunk >> 8) & 255
Keeps the second byte:
01100001 = 97 = "a"

chunk & 255
Keeps the third byte:
01101110 = 110 = "n"

&63 makes sure to only consider the 6 bit value


if (byteIndex < outputLength)
just to make sure we only write the bytes if valid
```

## Process image

Gets the image data from base64 to bytes to raw data using jpeg decode.

```typescript
  const { data, width, height } = jpeg.decode(
    base64ToUint8Array(resized.base64),
    {
      useTArray: true,
    },
  );

  data = [
    R, G, B, A,
    R, G, B, A,
  ...]
```

allocate space for image batch

```typescript
  // allocate space for image batch, this should match what the model expects
  // should probably export it with classes and so...
  const floatData = new Float32Array(1 * 3 * width * height);
  [batch, channels, height, width]


  // loop on the image
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {

  // process one pixel at the time
  const idx = (y * width + x) * 4;


  // grab the rgb for current pixel
  const r = data[idx] / 255;
  const g = data[idx + 1] / 255;
  const b = data[idx + 2] / 255;

  // note: this needs to match the model normalization step

  // v2.Normalize(mean=[0.485 (r), 0.456 (g), 0.406 (b)], std=[0.229 (r), 0.224 (g), 0.225 (b)]),
  const r = (data[idx] / 255 - 0.485) / 0.229;
  const g = (data[idx + 1] / 255 - 0.456) / 0.224;
  const b = (data[idx + 2] / 255 - 0.406) / 0.225;

  // divide by 255 so that the value can be converted into something between 0 and 1
  // 255 because that is the max value 8 bits can represent

  // add the pixel to the float data CHW channel height width

  const pixelIndex = y * width + x;

  // R channel
  floatData[pixelIndex] = r;

  // G channel
  floatData[width * height + pixelIndex] = g;

  // B channel
  floatData[2 * width * height + pixelIndex] = b;

  floatData[0] = red of pixel 0
  floatData[1] = red of pixel 1
  floatData[2] = red of pixel 2
  floatData[3] = red of pixel 3
  floatData[4] = green of pixel 0
  floatData[5] = green of pixel 1
  floatData[6] = green of pixel 2
  floatData[7] = green of pixel 3
  floatData[8] = blue of pixel 0
  floatData[9] = blue of pixel 1
  floatData[10] = blue of pixel 2
  floatData[11] = blue of pixel 3

  // returns a tensor with float data and expected model config
  return new Tensor("float32", floatData, [1, 3, height, width]);
```

## Run inference

```bash
 LOG  inputs ["x"]
 LOG  outputs ["linear_2"]
 LOG  {"linear_2": {"cpuData": [-0.10486273467540741, -0.07201680541038513, -0.03084525279700756, -0.03251670300960541, 0.0016371747478842735, 0.03331896290183067, 0.07929269969463348, 0.09253919124603271, -0.05170805752277374, 0.01252450980246067], "dataLocation": "cpu", "dims": [1, 10], "size": 10, "type": "float32"}}
```

This returns the logits the model predicts.

In order to find the top prediction use softmax on them:

```typescript
      const probabilities = softmax(logits);
      const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
      const confidence = probabilities[predictedIndex];

      // match against label from model data
```

TODO:
- add confidence threshold.
- download model, config and classes on app start if not present.
  - maybe host on huggingface

```bash
const MODEL_URL =
  "https://huggingface.co/name/trail-lens-model/resolve/main/model.onnx";
```