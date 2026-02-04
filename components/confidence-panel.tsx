"use client";

import { useEffect, useState } from "react";
import { Activity, TrendingUp, Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface ConfidencePanelProps {
  confidence: number;
  isProcessing: boolean;
}

export function ConfidencePanel({
  confidence,
  isProcessing,
}: ConfidencePanelProps) {
  const [displayConfidence, setDisplayConfidence] = useState(0);

  // Direct update without artificial delay
  useEffect(() => {
    setDisplayConfidence(confidence);
  }, [confidence]);

  const getConfidenceColor = (value: number) => {
    if (value >= 90) return "text-green-400";
    if (value >= 75) return "text-gradient-2";
    if (value >= 50) return "text-yellow-400";
    return "text-red-400";
  };

  const getConfidenceLabel = (value: number) => {
    if (value >= 90) return "Excellent";
    if (value >= 75) return "Good";
    if (value >= 50) return "Fair";
    if (value > 0) return "Low";
    return "—";
  };

  return (
    <div className="relative overflow-hidden rounded-2xl border border-border/50 bg-card/80 p-1 shadow-xl shadow-gradient-2/5 backdrop-blur-xl">
      {/* Gradient border effect */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-bl from-gradient-2/30 via-transparent to-gradient-3/30 opacity-50" />

      <div className="relative rounded-xl bg-background/90 p-6">
        {/* Header */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="rounded-lg bg-gradient-2/10 p-2">
              <Activity className="h-5 w-5 text-gradient-2" />
            </div>
            <h2 className="font-mono text-sm font-medium text-muted-foreground">
              CONFIDENCE
            </h2>
          </div>

          {/* Confidence label badge */}
          {!isProcessing && displayConfidence > 0 && (
            <div className="flex items-center gap-1 rounded-full bg-secondary/50 px-3 py-1">
              <TrendingUp
                className={`h-3 w-3 ${getConfidenceColor(displayConfidence)}`}
              />
              <span
                className={`text-xs font-medium ${getConfidenceColor(displayConfidence)}`}
              >
                {getConfidenceLabel(displayConfidence)}
              </span>
            </div>
          )}
        </div>

        {/* Confidence Score */}
        <div className="mb-4 flex items-baseline gap-1">
          {isProcessing ? (
            <div className="flex items-center gap-3">
              <Loader2 className="h-6 w-6 animate-spin text-gradient-2" />
              <span className="text-xl text-muted-foreground">
                Calculating...
              </span>
            </div>
          ) : (
            <>
              <span
                className={`text-5xl font-bold tabular-nums ${getConfidenceColor(displayConfidence)}`}
              >
                {displayConfidence}
              </span>
              <span className="text-2xl text-muted-foreground">%</span>
            </>
          )}
        </div>

        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="relative h-3 overflow-hidden rounded-full bg-secondary/50">
            {/* Animated gradient background */}
            <div
              className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-gradient-1 via-gradient-2 to-gradient-3 transition-all duration-500 ease-out"
              style={{ width: `${displayConfidence}%` }}
            />

            {/* Shine effect */}
            <div
              className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-transparent via-white/20 to-transparent transition-all duration-500 ease-out"
              style={{ width: `${displayConfidence}%` }}
            />
          </div>

          {/* Scale markers */}
          <div className="flex justify-between text-xs text-muted-foreground/50">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
