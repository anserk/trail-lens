import { ImageManipulator, SaveFormat } from "expo-image-manipulator";
import jpeg from "jpeg-js";
import { Tensor } from "onnxruntime-react-native";

const base64Alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const base64ToUint8Array = (base64: string) => {
  const sanitized = base64.replace(/\s/g, "");
  const outputLength =
    Math.floor((sanitized.length * 3) / 4) -
    (sanitized.endsWith("==") ? 2 : sanitized.endsWith("=") ? 1 : 0);
  const bytes = new Uint8Array(outputLength);

  let byteIndex = 0;

  for (let i = 0; i < sanitized.length; i += 4) {
    const chunk =
      (base64Alphabet.indexOf(sanitized[i]) << 18) |
      (base64Alphabet.indexOf(sanitized[i + 1]) << 12) |
      ((base64Alphabet.indexOf(sanitized[i + 2]) & 63) << 6) |
      (base64Alphabet.indexOf(sanitized[i + 3]) & 63);

    if (byteIndex < outputLength) bytes[byteIndex++] = (chunk >> 16) & 255;
    if (byteIndex < outputLength) bytes[byteIndex++] = (chunk >> 8) & 255;
    if (byteIndex < outputLength) bytes[byteIndex++] = chunk & 255;
  }

  return bytes;
};

export const preprocessImage = async (uri: string) => {
  // Resize image
  const image = await ImageManipulator.manipulate(uri)
    .resize({ width: 224, height: 224 })
    .renderAsync();

  const resized = await image.saveAsync({
    compress: 1,
    format: SaveFormat.JPEG,
    base64: true,
  });

  if (!resized.base64) {
    throw new Error("Failed to get base64 image");
  }

  // Decode jpeg
  const { data, width, height } = jpeg.decode(
    base64ToUint8Array(resized.base64),
    {
      useTArray: true,
    },
  );

  // allocate space for image batch, this should match what the model expect
  // should probably export it with classes and so...
  const floatData = new Float32Array(1 * 3 * width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;

      // Need to apply same normalize as the training step, again something else
      // that needs to be kept in sync with the model.
      // v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      const r = (data[idx] / 255 - 0.485) / 0.229;
      const g = (data[idx + 1] / 255 - 0.456) / 0.224;
      const b = (data[idx + 2] / 255 - 0.406) / 0.225;

      const pixelIndex = y * width + x;

      // R channel
      floatData[pixelIndex] = r;

      // G channel
      floatData[width * height + pixelIndex] = g;

      // B channel
      floatData[2 * width * height + pixelIndex] = b;
    }
  }

  return new Tensor("float32", floatData, [1, 3, width, height]);
};
