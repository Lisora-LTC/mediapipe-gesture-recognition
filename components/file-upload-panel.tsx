"use client";

import React from "react";

import { useState, useRef } from "react";
import { Upload, ImageIcon, X, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FileUploadPanelProps {
  onFileAnalyze: (file: File) => void;
  isProcessing: boolean;
}

export function FileUploadPanel({
  onFileAnalyze,
  isProcessing,
}: FileUploadPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (file: File | null) => {
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      // Automatically analyze on selection
      onFileAnalyze(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileChange(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="flex flex-col items-center justify-center">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
        className="hidden"
        id="file-upload"
      />

      {preview ? (
        <div className="relative w-full max-w-md">
          {/* Preview Image */}
          <div className="relative overflow-hidden rounded-xl border border-border/50">
            <img
              src={preview || "/placeholder.svg"}
              alt="Preview"
              className="h-64 w-full object-contain bg-black/50"
            />

            {/* Processing overlay */}
            {isProcessing && (
              <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
                <div className="flex flex-col items-center gap-3">
                  <Loader2 className="h-8 w-8 animate-spin text-gradient-1" />
                  <span className="text-sm text-muted-foreground">
                    Analyzing gesture...
                  </span>
                </div>
              </div>
            )}

            {/* Clear button */}
            <button
              onClick={clearFile}
              className="absolute right-2 top-2 rounded-full bg-background/80 p-1.5 text-muted-foreground transition-colors hover:bg-background hover:text-foreground"
              type="button"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {/* Re-analyze button */}
          <Button
            onClick={() => selectedFile && onFileAnalyze(selectedFile)}
            disabled={isProcessing}
            className="mt-4 w-full bg-gradient-to-r from-gradient-1 to-gradient-2 text-primary-foreground hover:opacity-90"
          >
            {isProcessing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              "Analyze Again"
            )}
          </Button>
        </div>
      ) : (
        <label
          htmlFor="file-upload"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`group flex h-64 w-full max-w-md cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all duration-300 ${
            isDragging
              ? "border-gradient-1 bg-gradient-1/10"
              : "border-border/50 bg-secondary/30 hover:border-gradient-1/50 hover:bg-gradient-1/5"
          }`}
        >
          <div
            className={`mb-4 rounded-full p-4 transition-all duration-300 ${
              isDragging
                ? "bg-gradient-1/20"
                : "bg-secondary group-hover:bg-gradient-1/10"
            }`}
          >
            <Upload
              className={`h-8 w-8 transition-colors ${
                isDragging
                  ? "text-gradient-1"
                  : "text-muted-foreground group-hover:text-gradient-1"
              }`}
            />
          </div>
          <p className="mb-1 font-medium text-foreground">
            Drop your image here
          </p>
          <p className="text-sm text-muted-foreground">or click to browse</p>
          <p className="mt-4 text-xs text-muted-foreground/70">
            Supports: JPG, PNG, WebP
          </p>
        </label>
      )}
    </div>
  );
}
