import { preprocessImage } from "@/utilities/image";
import AntDesign from "@expo/vector-icons/AntDesign";
import Feather from "@expo/vector-icons/Feather";
import FontAwesome6 from "@expo/vector-icons/FontAwesome6";
import { Asset } from "expo-asset";
import {
  CameraMode,
  CameraType,
  CameraView,
  useCameraPermissions,
} from "expo-camera";
import { Image } from "expo-image";
import {
  InferenceSession
} from "onnxruntime-react-native";
import { useEffect, useRef, useState } from "react";
import { Button, Pressable, StyleSheet, Text, View } from "react-native";
import modelPath from "../../assets/model.onnx";

const CLASS_NAMES = [
  "poison ivy",
  "virginia creeper",
  "red maple",
  "white oak",
  "tulip poplar",
  "loblolly pine",
  "eastern red cedar",
  "american holly",
  "flowering dogwood",
  "black cherry",
]

function softmax(values: number[]) {
  const max = Math.max(...values);

  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);

  return exps.map((v) => v / sum);
}

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const ref = useRef<CameraView>(null);
  const [uri, setUri] = useState<string | null>(null);
  const [mode, setMode] = useState<CameraMode>("picture");
  const [facing, setFacing] = useState<CameraType>("back");
  // const [recording, setRecording] = useState(false);
  const [className, setClassName] = useState<null | string>(null);
  const [classNameProb, setClassNameProb] = useState<null | number>(null)

  const [session, setSession] =
    useState<InferenceSession | null>(null);

  useEffect(() => {
    async function loadModel() {
      const asset = Asset.fromModule(modelPath);

      await asset.downloadAsync();

      const loadedSession =
        await InferenceSession.create(
          asset.localUri ?? asset.uri
        );

      setSession(loadedSession);
    }

    loadModel().catch(console.error);
  }, []);

  if (!permission) {
    return null;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: "center" }}>
          We need your permission to use the camera
        </Text>
        <Button onPress={requestPermission} title="Grant permission" />
      </View>
    );
  }

  const takePicture = async () => {
    const photo = await ref.current?.takePictureAsync();
    setClassName(null)
    setClassNameProb(null)

    if (session !== null && photo != null) {
      const inputTensor = await preprocessImage(photo.uri);

      const results = await session.run({
        x: inputTensor,
      });

      const output = results["linear_2"];
      const logits = Array.from(output.data as Float32Array);

      const probabilities = softmax(logits);

      const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
      const confidence = probabilities[predictedIndex];
      const className = CLASS_NAMES[predictedIndex]
      setClassName(className)
      setClassNameProb(confidence)

    }

    if (photo?.uri) setUri(photo.uri);
  };

  // const recordVideo = async () => {
  //   if (recording) {
  //     setRecording(false);
  //     ref.current?.stopRecording();
  //     return;
  //   }
  //   setRecording(true);
  //   const video = await ref.current?.recordAsync();
  //   console.log({ video });
  // };

  const toggleMode = () => {
    setMode((prev) => (prev === "picture" ? "video" : "picture"));
  };

  const toggleFacing = () => {
    setFacing((prev) => (prev === "back" ? "front" : "back"));
  };

  const renderPicture = (uri: string) => {
    return (
      <View style={styles.cameraContainer}>
        <Image
          source={{ uri }}
          contentFit="cover"
          style={styles.camera}
        />
        <View style={styles.shutterContainer}>
          <Pressable onPress={() => setUri(null)}>
            <FontAwesome6 name="rotate-left" size={32} color="white" />
          </Pressable>
          {className && classNameProb &&
            <Text>{className}-{(classNameProb * 100).toFixed(2)}%</Text>}
        </View>
      </View>
    );
  };

  const renderCamera = () => {
    return (
      <View style={styles.cameraContainer}>
        <CameraView
          style={styles.camera}
          ref={ref}
          mode={mode}
          facing={facing}
          mute={false}
          responsiveOrientationWhenOrientationLocked
        />
        <View style={styles.shutterContainer}>
          <Pressable onPress={toggleMode}>
            {mode === "picture" ? (
              <AntDesign name="picture" size={32} color="white" />
            ) : (
              <Feather name="video" size={32} color="white" />
            )}
          </Pressable>
          {/* onPress={mode === "picture" ? takePicture : recordVideo} */}
          <Pressable onPress={takePicture}>
            {({ pressed }) => (
              <View
                style={[
                  styles.shutterBtn,
                  {
                    opacity: pressed ? 0.5 : 1,
                  },
                ]}
              >
                <View
                  style={[
                    styles.shutterBtnInner,
                    {
                      backgroundColor: mode === "picture" ? "white" : "red",
                    },
                  ]}
                />
              </View>
            )}
          </Pressable>
          <Pressable onPress={toggleFacing}>
            <FontAwesome6 name="rotate-left" size={32} color="white" />
          </Pressable>
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      {uri ? renderPicture(uri) : renderCamera()}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  cameraContainer: StyleSheet.absoluteFillObject,
  camera: StyleSheet.absoluteFillObject,
  shutterContainer: {
    position: "absolute",
    bottom: 44,
    left: 0,
    width: "100%",
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 30,
  },
  shutterBtn: {
    backgroundColor: "transparent",
    borderWidth: 5,
    borderColor: "white",
    width: 85,
    height: 85,
    borderRadius: 45,
    alignItems: "center",
    justifyContent: "center",
  },
  shutterBtnInner: {
    width: 70,
    height: 70,
    borderRadius: 50,
  },
});
