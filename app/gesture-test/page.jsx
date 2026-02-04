import GestureModel from "@/app/components/GestureModel";

export default function GestureTestPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-background text-foreground p-8 gap-8">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-extrabold tracking-tight lg:text-4xl text-gradient-1">
          Gesture Recognition Test
        </h1>
        <p className="text-muted-foreground max-w-[600px]">
          Testing the ONNX model integration with MediaPipe. Please allow camera
          access.
        </p>
      </div>

      <GestureModel />

      <div className="grid grid-cols-3 gap-4 text-sm text-center">
        <div className="p-4 border rounded bg-card/50">
          <div className="font-bold">Gesture 1</div>
          <div className="text-xs text-muted-foreground">Description...</div>
        </div>
        <div className="p-4 border rounded bg-card/50">
          <div className="font-bold">Gesture 2</div>
          <div className="text-xs text-muted-foreground">Description...</div>
        </div>
        <div className="p-4 border rounded bg-card/50">
          <div className="font-bold">Gesture 3</div>
          <div className="text-xs text-muted-foreground">...</div>
        </div>
        {/* Add more as needed */}
      </div>
    </div>
  );
}
