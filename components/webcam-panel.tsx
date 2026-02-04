"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { Camera, VideoOff, Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DrawingUtils, HandLandmarker } from "@mediapipe/tasks-vision";
import { Prediction } from "@/lib/gesture-service";

interface WebcamPanelProps {
  predictFn: (
    video: HTMLVideoElement,
    startTime: number,
  ) => Promise<Prediction | null>;
  onResult: (result: Prediction) => void;
  isReady: boolean;
}

const GESTURE_CLASSES: Record<number, string> = {
  1: "Gesture 1",
  2: "Gesture 2",
  3: "Gesture 3",
  4: "Gesture 4",
  5: "Gesture 5",
  6: "Gesture 6",
};

export function WebcamPanelComponent({
  predictFn,
  onResult,
  isReady,
}: WebcamPanelProps) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);

  // Local stats state (High frequency)
  const [fps, setFps] = useState(0);
  const [localPred, setLocalPred] = useState<Prediction | null>(null);

  const lastUiUpdateRef = useRef<number>(0);

  const predictLoop = useCallback(
    async (startTime: number) => {
      if (videoRef.current && canvasRef.current && isStreaming && isReady) {
        const now = performance.now();

        // Only process if video time has advanced
        if (videoRef.current.currentTime !== lastTimeRef.current) {
          lastTimeRef.current = videoRef.current.currentTime;

          const result = await predictFn(videoRef.current, now);

          const canvas = canvasRef.current;
          const ctx = canvas.getContext("2d");

          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (result) {
              // Draw Landmarks (Always draw for smoothness)
              if (result.landmarks) {
                const drawingUtils = new DrawingUtils(ctx);
                drawingUtils.drawConnectors(
                  result.landmarks,
                  HandLandmarker.HAND_CONNECTIONS,
                  { color: "#00FF00", lineWidth: 2 },
                );
                drawingUtils.drawLandmarks(result.landmarks, {
                  color: "#FF0000",
                  lineWidth: 1,
                  radius: 2,
                });
              }

              // Throttle UI updates (FPS & Stats) to avoid React render spam
              // Update UI every 100ms (10fps for UI is enough)
              if (now - lastUiUpdateRef.current > 100) {
                // Calc FPS based on this frame delta (instantaneous)
                const delta = now - startTime;
                if (delta > 0) {
                  setFps(Math.round(1000 / delta));
                }

                setLocalPred(result);
                // Propagate to parent (Parent should debounce this if needed)
                onResult(result);

                lastUiUpdateRef.current = now;
              }
            } else {
              // Clear if no result
              if (now - lastUiUpdateRef.current > 100) {
                setLocalPred(null);
                onResult({ label: null, prob: 0, stable: false });
                lastUiUpdateRef.current = now;
              }
            }
          }
        }

        requestRef.current = requestAnimationFrame(() => predictLoop(now));
      } else if (isStreaming && isReady) {
        // Keep looping even if conditions not momentarily met
        requestRef.current = requestAnimationFrame(() =>
          predictLoop(performance.now()),
        );
      }
    },
    [isStreaming, isReady, predictFn, onResult],
  );

  // Start/Stop loop based on streaming state
  useEffect(() => {
    if (isStreaming && isReady) {
      const startTime = performance.now();
      requestRef.current = requestAnimationFrame(() => predictLoop(startTime));
    } else {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isStreaming, isReady, predictLoop]);

  const startWebcam = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch {
      setError("Unable to access camera. Please grant permission.");
      setIsStreaming(false);
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setLocalPred(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, [stopWebcam]);

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative w-full max-w-md overflow-hidden rounded-xl border border-border/50 bg-black/50">
        {/* Video element */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`h-64 w-full object-cover -scale-x-100 ${!isStreaming && "hidden"}`}
        />

        {/* Canvas Overlay */}
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className={`absolute top-0 left-0 w-full h-full object-cover pointer-events-none -scale-x-100 ${!isStreaming && "hidden"}`}
        />

        {/* Stats Overlay (Glassmorphism) */}
        {isStreaming && (
          <div className="absolute top-4 left-4 bg-black/60 text-white p-3 rounded-md backdrop-blur-sm z-10 transition-all duration-300 border border-white/10 pointer-events-none">
            <div className="text-xs text-gray-300 uppercase tracking-wider mb-1">
              Prediction
            </div>
            {localPred && localPred.label !== null ? (
              <>
                <div
                  className={`text-xl font-bold ${localPred.stable ? "text-green-400" : "text-yellow-400"}`}
                >
                  {localPred.stable
                    ? GESTURE_CLASSES[localPred.label]
                    : "Stabilizing..."}
                </div>
                <div className="text-xs text-gray-400 mt-1 font-mono">
                  Conf: {Math.round(localPred.prob * 100)}% | FPS: {fps}
                </div>
              </>
            ) : (
              <div className="text-lg text-gray-500 italic">Not detected</div>
            )}
          </div>
        )}

        {/* Placeholder when not streaming */}
        {!isStreaming && (
          <div className="flex h-64 flex-col items-center justify-center gap-4">
            {error ? (
              <>
                <div className="rounded-full bg-destructive/10 p-4">
                  <VideoOff className="h-8 w-8 text-destructive" />
                </div>
                <p className="text-sm text-destructive">{error}</p>
                <Button
                  onClick={startWebcam}
                  variant="outline"
                  size="sm"
                  className="border-border/50 bg-secondary/50 text-foreground"
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Retry
                </Button>
              </>
            ) : (
              <>
                <div className="rounded-full bg-gradient-1/10 p-4">
                  <Camera className="h-8 w-8 text-gradient-1" />
                </div>
                <p className="text-sm text-muted-foreground">Camera is off</p>
                <Button
                  onClick={startWebcam}
                  className="bg-gradient-to-r from-gradient-1 to-gradient-2 text-primary-foreground hover:opacity-90"
                >
                  Enable Camera
                </Button>
              </>
            )}
          </div>
        )}

        {/* Loading overlay */}
        {isStreaming && !isReady && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm z-20">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="h-8 w-8 animate-spin text-gradient-1" />
              <span className="text-sm text-muted-foreground">
                Initializing Model...
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="mt-4 flex gap-3">
        {isStreaming ? (
          <Button
            onClick={stopWebcam}
            variant="outline"
            className="border-border/50 bg-secondary/50 text-foreground"
          >
            <VideoOff className="mr-2 h-4 w-4" />
            Stop Camera
          </Button>
        ) : (
          !error && (
            <p className="text-sm text-muted-foreground">
              Click Enable Camera to start real-time recognition
            </p>
          )
        )}
      </div>
    </div>
  );
}

export const WebcamPanel = React.memo(WebcamPanelComponent);
