"use client";

import { Hand, Loader2 } from "lucide-react";

interface ResultsPanelProps {
  gestureResult: string;
  isProcessing: boolean;
}

export function ResultsPanel({
  gestureResult,
  isProcessing,
}: ResultsPanelProps) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-border/50 bg-card/80 p-1 shadow-xl shadow-gradient-1/5 backdrop-blur-xl">
      {/* Gradient border effect */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-gradient-1/30 via-transparent to-gradient-2/30 opacity-50" />

      <div className="relative rounded-xl bg-background/90 p-6">
        {/* Header */}
        <div className="mb-4 flex items-center gap-2">
          <div className="rounded-lg bg-gradient-1/10 p-2">
            <Hand className="h-5 w-5 text-gradient-1" />
          </div>
          <h2 className="font-mono text-sm font-medium text-muted-foreground">
            DETECTED GESTURE
          </h2>
        </div>

        {/* Result */}
        <div className="flex min-h-20 items-center">
          {isProcessing ? (
            <div className="flex items-center gap-3">
              <Loader2 className="h-6 w-6 animate-spin text-gradient-1" />
              <span className="text-xl text-muted-foreground">
                Analyzing...
              </span>
            </div>
          ) : (
            <h3 className="bg-gradient-to-r from-gradient-1 via-gradient-2 to-gradient-3 bg-clip-text text-4xl font-bold tracking-tight text-transparent md:text-5xl">
              {gestureResult}
            </h3>
          )}
        </div>

        {/* Decorative element */}
        <div className="mt-4 h-1 w-24 rounded-full bg-gradient-to-r from-gradient-1 to-gradient-2 opacity-50" />
      </div>
    </div>
  );
}
