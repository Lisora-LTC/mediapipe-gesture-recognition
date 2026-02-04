import * as ort from "onnxruntime-web";
import {
  FilesetResolver,
  HandLandmarker,
  DrawingUtils,
  HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

// Constants
// Append timestamp to force reload the model (bust cache)
// You can remove this in production
const ONNX_MODEL_PATH = "/gesture_model.onnx?v=" + new Date().getTime();
const HAND_LANDMARKER_TASK =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

export interface Prediction {
  label: number | null;
  prob: number;
  stable: boolean;
  landmarks?: any;
}

// Helper: Calculate Euclidean Distance
function getDistance(p1: any, p2: any) {
  return Math.sqrt(
    Math.pow(p1.x - p2.x, 2) +
      Math.pow(p1.y - p2.y, 2) +
      Math.pow(p1.z - p2.z, 2),
  );
}

// Helper: Build Features (23 dims)
function buildGestureFeatures(landmarks: any[]) {
  if (!landmarks || landmarks.length !== 21) return null;

  // 1. Base length: Wrist(0) to Middle Finger Tip(12)
  let baseLen = getDistance(landmarks[0], landmarks[12]);
  baseLen = Math.max(baseLen, 1e-6);

  const features: number[] = [];

  // 2. Relative Distances (15 pairs)
  const distancePairs = [
    [4, 8],
    [8, 12],
    [12, 16],
    [16, 20],
    [4, 20], // Fingertips
    [0, 4],
    [0, 8],
    [0, 12],
    [0, 16],
    [0, 20], // Wrist to Fingertips
    [8, 5],
    [5, 6],
    [6, 7],
    [12, 9],
    [20, 17], // Knuckles
  ];

  for (const [p1, p2] of distancePairs) {
    const dist = getDistance(landmarks[p1], landmarks[p2]);
    features.push(dist / baseLen);
  }

  // 3. Angles (8 triples)
  const calcAngle = (pA: any, pB: any, pC: any) => {
    const ba = { x: pA.x - pB.x, y: pA.y - pB.y, z: pA.z - pB.z };
    const bc = { x: pC.x - pB.x, y: pC.y - pB.y, z: pC.z - pB.z };

    const dotProduct = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
    const lenBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y + ba.z * ba.z);
    const lenBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z);

    const denominator = lenBA * lenBC + 1e-6;
    let cosAng = dotProduct / denominator;

    if (cosAng > 1.0) cosAng = 1.0;
    if (cosAng < -1.0) cosAng = -1.0;

    const angleRad = Math.acos(cosAng);
    return angleRad * (180.0 / Math.PI);
  };

  const angleTriples = [
    [4, 0, 8],
    [8, 0, 12],
    [12, 0, 16],
    [16, 0, 20],
    [0, 5, 8],
    [0, 9, 12],
    [0, 13, 16],
    [0, 17, 20],
  ];

  for (const [a, b, c] of angleTriples) {
    features.push(calcAngle(landmarks[a], landmarks[b], landmarks[c]));
  }

  return Float32Array.from(features);
}

// Class: Stabilizer
export class GestureStabilizer {
  windowSize: number;
  threshold: number;
  history: Array<{ label: number; prob: number }>;

  constructor(windowSize = 8, threshold = 0.6) {
    this.windowSize = windowSize;
    this.threshold = threshold;
    this.history = [];
  }

  update(label: number, prob: number) {
    this.history.push({ label, prob });
    if (this.history.length > this.windowSize) {
      this.history.shift();
    }

    if (this.history.length < this.windowSize) {
      return { label, prob, stable: false };
    }

    // Vote
    const counts: { [key: number]: number } = {};
    for (const item of this.history) {
      counts[item.label] = (counts[item.label] || 0) + 1;
    }

    let maxLabel = label;
    let maxCount = 0;
    for (const [lbl, cnt] of Object.entries(counts)) {
      if (cnt > maxCount) {
        maxCount = cnt;
        maxLabel = parseInt(lbl);
      }
    }

    const sameLabelItems = this.history.filter(
      (item) => item.label == maxLabel,
    );
    const avgProb =
      sameLabelItems.reduce((sum, item) => sum + item.prob, 0) /
      sameLabelItems.length;

    if (maxCount >= this.windowSize * 0.6 && avgProb > this.threshold) {
      return { label: maxLabel, prob: avgProb, stable: true };
    } else {
      // Return maxLabel even if unstable, but mark stable as false
      return { label: maxLabel, prob: avgProb, stable: false };
    }
  }

  reset() {
    this.history = [];
  }
}

// Logic: GestureService
class GestureService {
  private handLandmarker: HandLandmarker | null = null;
  private onnxSession: ort.InferenceSession | null = null;
  private stabilizer: GestureStabilizer;
  private isLoaded: boolean = false;

  private currentRunningMode: "IMAGE" | "VIDEO" | null = null;

  constructor() {
    this.stabilizer = new GestureStabilizer(5, 0.7); // Tighter window for faster feeling
  }

  async initialize() {
    if (this.isLoaded) return;

    // Suppress ONNX warnings
    ort.env.logLevel = "error";

    console.log("Loading GestureService components...");

    // 1. MediaPipe
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_LANDMARKER_TASK,
        delegate: "GPU",
      },
      runningMode: "VIDEO", // Default to VIDEO
      numHands: 1,
    });
    this.currentRunningMode = "VIDEO";

    // 2. ONNX
    this.onnxSession = await ort.InferenceSession.create(ONNX_MODEL_PATH, {
      executionProviders: ["wasm"],
    });

    this.isLoaded = true;
    console.log("GestureService initialized.");
  }

  // Predict from Video Stream
  async predictVideo(videoElement: HTMLVideoElement, startTimeMs: number) {
    if (!this.isLoaded || !this.handLandmarker || !this.onnxSession) {
      throw new Error("Service not initialized");
    }

    // Ensure running mode is VIDEO
    if (this.currentRunningMode !== "VIDEO" && this.handLandmarker.setOptions) {
      await this.handLandmarker.setOptions({ runningMode: "VIDEO" });
      this.currentRunningMode = "VIDEO";
    }

    // Ensure valid video dimensions (Fix for "ROI width > 0" error)
    if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
      // console.warn("Skipping prediction: Video dimensions are 0");
      return { label: null, prob: 0, stable: false };
    }

    const startTime = performance.now();
    const results = this.handLandmarker.detectForVideo(
      videoElement,
      startTimeMs,
    );
    const result = await this._processResults(results, true);
    const endTime = performance.now();
    // Log every ~100 frames or so to not spam, or just log
    if (Math.random() < 0.05) {
      console.log(
        `Predict + Process time: ${Math.round(endTime - startTime)}ms`,
      );
    }
    return result;
  }

  // Predict from Static Image
  async predictImage(imageElement: HTMLImageElement) {
    if (!this.isLoaded || !this.handLandmarker || !this.onnxSession) {
      throw new Error("Service not initialized");
    }

    // Switch to IMAGE mode
    if (this.currentRunningMode !== "IMAGE" && this.handLandmarker.setOptions) {
      console.log("Switching to IMAGE mode...");
      await this.handLandmarker.setOptions({ runningMode: "IMAGE" });
      this.currentRunningMode = "IMAGE";
    }

    console.log(
      `Predicting Image: ${imageElement.width}x${imageElement.height} (Natural: ${imageElement.naturalWidth}x${imageElement.naturalHeight})`,
    );

    try {
      const results = this.handLandmarker.detect(imageElement);
      console.log("Image Detection Results:", results);
      return await this._processResults(results, false); // No stabilization for static images
    } catch (e) {
      console.error("Image detection failed:", e);
      return null;
    }
  }

  private async _processResults(
    results: HandLandmarkerResult,
    useStabilizer: boolean,
  ) {
    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      const features = buildGestureFeatures(landmarks);

      if (features) {
        const tensor = new ort.Tensor("float32", features, [1, 23]);
        const feeds = { [this.onnxSession!.inputNames[0]]: tensor };

        const labelName = this.onnxSession!.outputNames[0];
        let predClass = -1;
        let prob = 1.0;

        // Robust Fetch Strategy
        try {
          // 1. Fetch output (Optimistic)
          // With zipmap=False, output[1] is a Float32 Tensor [1, n_classes]
          const fetches = [...this.onnxSession!.outputNames];
          const output = await this.onnxSession!.run(feeds, fetches);

          // Get Label
          predClass = Number(output[labelName].data[0]);

          // Get Probability
          const probName = this.onnxSession!.outputNames[1];
          if (probName && output[probName]) {
            const probTensor = output[probName];
            // Access underlying typed array
            const data = probTensor.data;

            if (data && data.length > 0) {
              // DEBUG: Check raw values
              if (Math.random() < 0.01) console.log("Raw ONNX Probs:", data);

              // Find max probability
              // Note: data might be Float32Array or standard array
              // @ts-ignore
              let maxVal = -Infinity;
              // @ts-ignore
              for (let i = 0; i < data.length; i++) {
                // @ts-ignore
                if (data[i] > maxVal) maxVal = data[i];
              }
              prob = maxVal;
            }
          }
        } catch (e) {
          console.warn("⚠️ Inference error (Safe Fallback):", e);
          const output = await this.onnxSession!.run(feeds, [labelName]);
          predClass = Number(output[labelName].data[0]);
          prob = 1.0;
        }

        if (useStabilizer) {
          const stabilized = this.stabilizer.update(predClass, prob);
          return { ...stabilized, landmarks };
        } else {
          return { label: predClass, prob, stable: true, landmarks };
        }
      }
    }
    return { label: null, prob: 0, stable: false };
  }
}

export const gestureService = new GestureService();
