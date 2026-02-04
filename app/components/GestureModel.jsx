"use client";

import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import {
  FilesetResolver,
  HandLandmarker,
  DrawingUtils,
} from "@mediapipe/tasks-vision";
import { Loader2, Camera, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

// === 配置 ===
const ONNX_MODEL_PATH = "/gesture_model.onnx";
// 根据你的网络情况，可能需要调整 wasm 路径，或者让其自动从 CDN 加载
// ort.env.wasm.wasmPaths = "/";

const GESTURE_CLASSES = {
  1: "Gesture 1",
  2: "Gesture 2",
  3: "Gesture 3",
  4: "Gesture 4",
  5: "Gesture 5",
  6: "Gesture 6",
};

// Suppress warnings
ort.env.logLevel = "error";

// === 辅助函数：计算欧氏距离 ===
function getDistance(p1, p2) {
  return Math.sqrt(
    Math.pow(p1.x - p2.x, 2) +
      Math.pow(p1.y - p2.y, 2) +
      Math.pow(p1.z - p2.z, 2),
  );
}

// === 核心逻辑：移植 Python 的 build_gesture_features ===
function buildGestureFeatures(landmarks) {
  // landmarks: Array of {x, y, z}

  if (!landmarks || landmarks.length !== 21) return null;

  // 1. 基准长度：腕部(0) 到 中指尖(12)
  let baseLen = getDistance(landmarks[0], landmarks[12]);
  baseLen = Math.max(baseLen, 1e-6);

  const features = [];

  // 2. 相对距离 (15个)
  const distancePairs = [
    [4, 8],
    [8, 12],
    [12, 16],
    [16, 20],
    [4, 20], // 指尖间
    [0, 4],
    [0, 8],
    [0, 12],
    [0, 16],
    [0, 20], // 腕部到指尖
    [8, 5],
    [5, 6],
    [6, 7],
    [12, 9],
    [20, 17], // 指节间
  ];

  for (const [p1, p2] of distancePairs) {
    const dist = getDistance(landmarks[p1], landmarks[p2]);
    features.push(dist / baseLen);
  }

  // 3. 角度计算 (8个)
  // helper to calc angle between vector BA and BC
  const calcAngle = (pA, pB, pC) => {
    // Vectors
    const ba = { x: pA.x - pB.x, y: pA.y - pB.y, z: pA.z - pB.z };
    const bc = { x: pC.x - pB.x, y: pC.y - pB.y, z: pC.z - pB.z };

    const dotProduct = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
    const lenBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y + ba.z * ba.z);
    const lenBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z);

    const denominator = lenBA * lenBC + 1e-6;
    let cosAng = dotProduct / denominator;

    // clip (-1, 1)
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

  // 返回 Float32Array，用于 ONNX Tensor
  return Float32Array.from(features);
}

// === 平滑逻辑 (Queue & Voting) ===
class GestureStabilizer {
  constructor(windowSize = 8, threshold = 0.6) {
    this.windowSize = windowSize;
    this.threshold = threshold;
    this.history = []; // Array of {label, prob}
  }

  update(label, prob) {
    this.history.push({ label, prob });
    if (this.history.length > this.windowSize) {
      this.history.shift();
    }

    if (this.history.length < this.windowSize) {
      return { label, prob };
    }

    // 统计出现次数
    const counts = {};
    for (const item of this.history) {
      counts[item.label] = (counts[item.label] || 0) + 1;
    }

    // 找到出现最多的 label
    let maxLabel = label;
    let maxCount = 0;
    for (const [lbl, cnt] of Object.entries(counts)) {
      if (cnt > maxCount) {
        maxCount = cnt;
        maxLabel = parseInt(lbl); // keys are strings
      }
    }

    // 计算平均置信度
    const sameLabelItems = this.history.filter(
      (item) => item.label == maxLabel,
    );
    const avgProb =
      sameLabelItems.reduce((sum, item) => sum + item.prob, 0) /
      sameLabelItems.length;

    // 投票阈值检查 (e.g. > 60% agreement)
    if (maxCount >= this.windowSize * 0.6 && avgProb > this.threshold) {
      return { label: maxLabel, prob: avgProb, stable: true };
    } else {
      return { label: "Unstable", prob: avgProb, stable: false };
    }
  }
}

export default function GestureModel() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState({
    label: null,
    prob: 0,
    stable: false,
  });
  const [fps, setFps] = useState(0);

  // References to keep objects alive without re-rendering
  const handLandmarkerRef = useRef(null);
  const onnxSessionRef = useRef(null);
  const stabilizerRef = useRef(new GestureStabilizer(8, 0.7));
  const requestRef = useRef(null);
  const lastTimeRef = useRef(0);

  useEffect(() => {
    async function init() {
      try {
        console.log("🚀 Starting initialization...");

        // 1. Load ONNX Model
        console.log("Loading ONNX model...");
        try {
          // Creating session
          const session = await ort.InferenceSession.create(ONNX_MODEL_PATH, {
            executionProviders: ["wasm"], // ensure wasm backend
          });
          onnxSessionRef.current = session;
          console.log(
            "✅ ONNX model loaded:",
            session.inputNames,
            session.outputNames,
          );
        } catch (e) {
          console.error("ONNX Load Failed:", e);
          throw new Error(
            `Failed to load ONNX model: ${e.message}. Ensure '${ONNX_MODEL_PATH}' exists in public folder.`,
          );
        }

        // 2. Load MediaPipe HandLandmarker
        console.log("Loading MediaPipe...");
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
        );

        handLandmarkerRef.current = await HandLandmarker.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numHands: 1,
          },
        );
        console.log("✅ MediaPipe loaded");

        // 3. Setup Camera
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.addEventListener("loadeddata", predictWebcam);
        }

        setLoading(false);
      } catch (err) {
        console.error("Initialization error:", err);
        setError(err.message);
        setLoading(false);
      }
    }

    init();

    return () => {
      // Cleanup
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
      if (handLandmarkerRef.current) handLandmarkerRef.current.close();
      // ONNX session doesn't strictly need explicit close in JS, gc handles it usually
    };
  }, []);

  const predictWebcam = async () => {
    if (
      !handLandmarkerRef.current ||
      !videoRef.current ||
      !onnxSessionRef.current
    )
      return;

    let startTimeMs = performance.now();

    // Detect Landmarks
    if (lastTimeRef.current !== videoRef.current.currentTime) {
      lastTimeRef.current = videoRef.current.currentTime;

      const results = handLandmarkerRef.current.detectForVideo(
        videoRef.current,
        startTimeMs,
      );

      // Draw on canvas
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (results.landmarks && results.landmarks.length > 0) {
        // We only care about the first hand
        const landmarks = results.landmarks[0]; // Array of {x, y, z} (normalized 0-1)

        // Drawing
        const drawingUtils = new DrawingUtils(ctx);
        drawingUtils.drawConnectors(
          landmarks,
          HandLandmarker.HAND_CONNECTIONS,
          { color: "#00FF00", lineWidth: 2 },
        );
        drawingUtils.drawLandmarks(landmarks, {
          color: "#FF0000",
          lineWidth: 1,
          radius: 2,
        });

        // === Feature Extraction & Inference ===
        try {
          // 1. Build features (23 dims)
          const features = buildGestureFeatures(landmarks);

          if (features) {
            // 2. Prepare Tensor
            // Shape: [1, 23]
            const tensor = new ort.Tensor("float32", features, [1, 23]);

            const feeds = {};
            // Note: You must ensure the input name matches your ONNX model!
            // 'float_input' matches the name we set in convert_to_onnx.py
            feeds[onnxSessionRef.current.inputNames[0]] = tensor;

            // 3. Run Inference - Fetch ONLY the label (index 0)
            // The second output (probabilities) is a Sequence<Map> which often causes
            // "Reading data from non-tensor typed value" errors in onnxruntime-web.
            const labelOutputName = onnxSessionRef.current.outputNames[0];
            const fetches = [labelOutputName];

            const results = await onnxSessionRef.current.run(feeds, fetches);

            // 4. Output processing
            const outputTensor = results[labelOutputName];

            // Ensure we handle BigInt or Number correctly
            // Onnxruntime-web might return BigInt64Array for Int64 labels
            let rawPred = outputTensor.data[0];
            const predClass = Number(rawPred);

            // Debug log for first success (optional, or keeps spamming?)
            console.log("Prediction:", predClass);

            // For probabilities, it's a bit more complex with ONNX Maps,
            // but for now let's just use the label or try to find prob map.
            // For simplicity, we assume robust prediction or just use 1.0 prob.
            // (Parsing ONNX Convert_sklearn Sequence<Map> output in JS is tricky,
            const prob = 1.0;

            // 5. Stabilizer
            const stableResult = stabilizerRef.current.update(predClass, prob);
            setPrediction(stableResult);
          }
        } catch (inferErr) {
          console.error("Inference Error:", inferErr);
        }
      } else {
        // No hand detected
        setPrediction({ label: null, prob: 0, stable: false });
      }
    }

    // Calc FPS
    const endTimeMs = performance.now();
    const delta = endTimeMs - startTimeMs;
    setFps(Math.round(1000 / Math.max(delta, 1)));

    requestRef.current = requestAnimationFrame(predictWebcam);
  };

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-destructive bg-destructive/10 rounded-xl">
        <AlertCircle className="w-10 h-10 mb-2" />
        <p className="font-semibold">Error Loading Model</p>
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  return (
    <div className="relative flex flex-col items-center gap-4 p-4 border rounded-xl bg-card">
      <div className="relative overflow-hidden rounded-lg shadow-lg border border-border">
        {/* Video & Canvas Overlay */}
        <video
          ref={videoRef}
          className="w-[640px] h-[480px] object-cover bg-black"
          autoPlay
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full"
          width={640}
          height={480}
        />

        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 text-white z-20">
            <Loader2 className="w-10 h-10 animate-spin mb-2 text-primary" />
            <p>Loading Model & MediaPipe...</p>
          </div>
        )}

        {/* Result Overlay */}
        {!loading && (
          <div className="absolute top-4 left-4 bg-black/60 text-white p-3 rounded-md backdrop-blur-sm z-10 transition-all duration-300">
            <div className="text-xs text-gray-300 uppercase tracking-wider mb-1">
              Prediction
            </div>
            {prediction.label !== null ? (
              <>
                <div
                  className={`text-2xl font-bold ${prediction.stable ? "text-green-400" : "text-yellow-400"}`}
                >
                  {prediction.stable
                    ? GESTURE_CLASSES[prediction.label]
                    : "Stabilizing..."}
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  Raw: {prediction.label} | FPS: {fps}
                </div>
              </>
            ) : (
              <div className="text-xl text-gray-500 italic">
                Not detected
              </div>
            )}
          </div>
        )}
      </div>

      <div className="text-sm text-muted-foreground max-w-lg text-center">
        Currently recognizing gestures 1-6. Ensure your hand is visible and
        well-lit.
      </div>
    </div>
  );
}
