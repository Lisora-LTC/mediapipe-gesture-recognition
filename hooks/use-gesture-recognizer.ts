import { useState, useEffect, useRef, useCallback } from "react";
import { gestureService, Prediction } from "@/lib/gesture-service";

export function useGestureRecognizer() {
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    const init = async () => {
      try {
        await gestureService.initialize();
        if (mounted) setIsReady(true);
      } catch (err: any) {
        console.error("Gesture Service Init Error:", err);
        if (mounted) setError(err.message);
      }
    };

    init();

    return () => {
      mounted = false;
    };
  }, []);

  const predictVideo = useCallback(
    async (video: HTMLVideoElement, startTime: number) => {
      if (!isReady) return null;
      return await gestureService.predictVideo(video, startTime);
    },
    [isReady],
  );

  const predictImage = useCallback(
    async (image: HTMLImageElement) => {
      if (!isReady) return null;
      return await gestureService.predictImage(image);
    },
    [isReady],
  );

  return { isReady, error, predictVideo, predictImage };
}
