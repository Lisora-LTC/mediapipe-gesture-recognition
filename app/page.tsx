"use client";

import { useState, useCallback, useEffect } from "react";
import { Hand, Sparkles } from "lucide-react";
import { ModeToggle } from "@/components/mode-toggle";
import { FileUploadPanel } from "@/components/file-upload-panel";
import { WebcamPanel } from "@/components/webcam-panel";
import { ResultsPanel } from "@/components/results-panel";
import { ConfidencePanel } from "@/components/confidence-panel";
import { useGestureRecognizer } from "@/hooks/use-gesture-recognizer";

const GESTURE_MAP: Record<number, string> = {
  1: "Gesture 1",
  2: "Gesture 2",
  3: "Gesture 3",
  4: "Gesture 4",
  5: "Gesture 5",
  6: "Gesture 6",
};

export default function GestureRecognitionPage() {
  const [mode, setMode] = useState<"upload" | "webcam">("upload");
  const [gestureResult, setGestureResult] = useState<string>(
    "Waiting for input...",
  );
  const [confidence, setConfidence] = useState<number>(0);
  const [isProcessing, setIsProcessing] = useState(false); // Used for image processing state

  const { isReady, error, predictVideo, predictImage } = useGestureRecognizer();

  // Handle Real-time Video Prediction Result (Debounced update for UI panels)
  // Note: High-freq updates are handled inside WebcamPanel locally
  const handleVideoResult = useCallback((result: any) => {
    // Optimization: Only update state if result changes significantly
    if (result && result.label !== null) {
      if (result.stable) {
        const newLabel =
          GESTURE_MAP[result.label as number] || `Unknown (${result.label})`;
        const newConf = Math.round(result.prob * 100);

        setGestureResult((prev) => (prev === newLabel ? prev : newLabel));
        setConfidence((prev) =>
          Math.abs(prev - newConf) < 5 ? prev : newConf,
        ); // Only update if conf changes > 5%
      } else {
        setGestureResult((prev) =>
          prev === "Stabilizing..." ? prev : "Stabilizing...",
        );
        // update confidence even if stabilizing, but throttled
        const newConf = Math.round(result.prob * 100);
        setConfidence((prev) =>
          Math.abs(prev - newConf) < 5 ? prev : newConf,
        );
      }
    } else {
      setGestureResult((prev) =>
        prev === "Not detected" ? prev : "Not detected",
      );
      setConfidence((prev) => (prev === 0 ? prev : 0));
    }
  }, []);

  // Handle Static Image Prediction
  const handleImageSelect = useCallback(
    async (file: File) => {
      if (!isReady) return;
      setIsProcessing(true);

      try {
        // Create an image element to pass to MediaPipe
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        await new Promise((resolve) => {
          img.onload = resolve;
        });

        const result = await predictImage(img);

        if (result && result.label !== null) {
          setGestureResult(
            GESTURE_MAP[result.label as number] || `Unknown (${result.label})`,
          );
          setConfidence(Math.round(result.prob * 100));
        } else {
          setGestureResult("No hand detected in image");
          setConfidence(0);
        }
      } catch (err) {
        console.error(err);
        setGestureResult("Error analyzing image");
      } finally {
        setIsProcessing(false);
      }
    },
    [isReady, predictImage],
  );

  return (
    <main className="relative min-h-screen overflow-hidden bg-background">
      {/* Background gradient effect */}
      <div
        className="pointer-events-none fixed inset-0 opacity-60"
        style={{
          background: `radial-gradient(circle at center, var(--gradient-1) 0%, transparent 70%)`,
          opacity: 0.2,
        }}
      />

      {/* Animated glow effects */}
      <div className="pointer-events-none fixed left-1/4 top-1/4 h-96 w-96 animate-pulse rounded-full bg-gradient-1/20 blur-3xl" />
      <div
        className="pointer-events-none fixed right-1/4 bottom-1/4 h-96 w-96 animate-pulse rounded-full bg-gradient-2/20 blur-3xl"
        style={{ animationDelay: "1s" }}
      />

      {/* Grid overlay */}
      <div
        className="pointer-events-none fixed inset-0 opacity-10"
        style={{
          backgroundImage: `linear-gradient(var(--gradient-1) 1px, transparent 1px), linear-gradient(90deg, var(--gradient-1) 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
        }}
      />

      {/* Content */}
      <div className="relative z-10 flex min-h-screen flex-col px-6 py-8">
        {/* Header */}
        <header className="mb-8 flex items-center justify-center">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Hand className="h-8 w-8 text-gradient-1" />
              <Sparkles className="absolute -right-1 -top-1 h-4 w-4 text-gradient-2" />
            </div>
            <h1 className="bg-gradient-to-r from-gradient-1 via-gradient-2 to-gradient-3 bg-clip-text font-mono text-2xl font-bold tracking-tight text-transparent">
              GestureAI
            </h1>
          </div>
        </header>

        {/* Mode Toggle - Apple Style */}
        <ModeToggle mode={mode} setMode={setMode} />

        {/* Model Loading State */}
        {!isReady && !error && (
          <div className="text-center mt-4 text-muted-foreground animate-pulse">
            Initializing AI Models...
          </div>
        )}
        {error && (
          <div className="text-center mt-4 text-destructive">
            Error: {error}
          </div>
        )}

        {/* Main Content Area */}
        <div className="mx-auto mt-8 w-full max-w-4xl flex-1">
          <div className="relative overflow-hidden rounded-2xl border border-border/50 bg-card/80 p-1 shadow-2xl shadow-gradient-1/10 backdrop-blur-xl">
            {/* Inner glow border */}
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-b from-gradient-1/20 via-transparent to-gradient-2/20 opacity-50" />

            <div className="relative rounded-xl bg-background/90 p-6">
              {mode === "upload" ? (
                <FileUploadPanel
                  onFileAnalyze={handleImageSelect}
                  isProcessing={isProcessing}
                />
              ) : (
                <WebcamPanel
                  predictFn={predictVideo}
                  onResult={handleVideoResult}
                  isReady={isReady}
                />
              )}
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="mx-auto mt-8 grid w-full max-w-4xl grid-cols-1 gap-6 md:grid-cols-2">
          {/* Left: Recognition Result */}
          <ResultsPanel
            gestureResult={gestureResult}
            isProcessing={isProcessing}
          />

          {/* Right: Confidence Score */}
          <ConfidencePanel
            confidence={confidence}
            isProcessing={isProcessing}
          />
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center">
          <p className="font-mono text-xs text-muted-foreground">
            Powered by MediaPipe & ONNX Runtime
          </p>
        </footer>
      </div>
    </main>
  );
}
