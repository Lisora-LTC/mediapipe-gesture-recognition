"use client";

import { Upload, Camera } from "lucide-react";

interface ModeToggleProps {
  mode: "upload" | "webcam";
  setMode: (mode: "upload" | "webcam") => void;
}

export function ModeToggle({ mode, setMode }: ModeToggleProps) {
  return (
    <div className="flex justify-center">
      <div className="relative inline-flex rounded-full border border-border/50 bg-secondary/50 p-1 backdrop-blur-sm">
        {/* Sliding background */}
        <div
          className="absolute inset-y-1 rounded-full bg-gradient-to-r from-gradient-1 to-gradient-2 shadow-lg shadow-gradient-1/30 transition-all duration-300 ease-out"
          style={{
            left: mode === "upload" ? "4px" : "50%",
            width: "calc(50% - 4px)",
          }}
        />

        {/* Upload Option */}
        <button
          onClick={() => setMode("upload")}
          className={`relative z-10 flex items-center gap-2 rounded-full px-6 py-2 font-medium transition-colors duration-200 ${
            mode === "upload"
              ? "text-primary-foreground"
              : "text-muted-foreground hover:text-foreground"
          }`}
          type="button"
        >
          <Upload className="h-4 w-4" />
          <span className="text-sm">Upload</span>
        </button>

        {/* Webcam Option */}
        <button
          onClick={() => setMode("webcam")}
          className={`relative z-10 flex items-center gap-2 rounded-full px-6 py-2 font-medium transition-colors duration-200 ${
            mode === "webcam"
              ? "text-primary-foreground"
              : "text-muted-foreground hover:text-foreground"
          }`}
          type="button"
        >
          <Camera className="h-4 w-4" />
          <span className="text-sm">Webcam</span>
        </button>
      </div>
    </div>
  );
}
