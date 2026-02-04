# GestureAI - 手势识别 Web 应用

这是一个基于 Next.js 和 AI 技术的实时手势识别应用。它结合了 Google MediaPipe 的手部关键点检测和自定义的 SVM 分类器（通过 ONNX Runtime 运行），能够在浏览器端直接进行高效、实时的手势识别，无需将视频流上传至服务器。

## ✨ 主要功能

- **📷 实时摄像头识别**：
  - 打开摄像头即可实时捕捉手部动作。
  - 平滑的关键点绘制（骨架图）。
  - 实时显示 FPS（帧率）和置信度。
  - 优化的性能，支持 30fps+ 流畅体验。
- **🖼️ 静态图片分析**：
  - 支持上传本地图片（JPG/PNG/WebP）。
  - 自动识别图片中的手势并输出结果。
- **📊 精准数据展示**：
  - **手势结果**：显示当前识别出的手势类别（Gesture 1-6）。
  - **置信度 (Confidence)**：通过进度条直观展示模型对当前判断的把握程度。
- **🎨 现代化 UI**：
  - 基于 Tailwind CSS 的精美界面。
  - 流畅的动画和玻璃拟态效果。
  - 响应式设计，适配不同屏幕。

## 🛠️ 技术栈

- **框架**: [Next.js](https://nextjs.org/) (React)
- **样式**: [Tailwind CSS](https://tailwindcss.com/) + [Lucide React](https://lucide.dev/) (图标)
- **AI 模型**:
  - 特征提取: [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
  - 分类器: [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) (SVM Model)
- **语言**: TypeScript / Python (用于模型训练与导出)

## 🚀 快速开始

### 1. 安装依赖

```bash
npm install
```

### 2. 准备模型 (重要)

本项目依赖于 ONNX 格式的 SVM 模型。如果你修改了训练数据或 Python 代码，需要重新生成模型：

1.  确保你已安装 Python 依赖 (`scikit-learn`, `onnx`, `skl2onnx`, `joblib`, `numpy`).
    - _注意：请使用项目兼容的 scikit-learn 版本，或确保导出脚本正确处理了版本兼容性。_
2.  运行导出脚本：
    ```bash
    python mlp/convert_to_onnx.py
    ```
    该脚本已优化，会自动禁用 `zipmap` 以适配 Web 端。
3.  将生成的 `gesture_model.onnx` 移动到 `public/` 目录：
    - 脚本运行后，请手动将 `mlp/gesture_model.onnx` 复制/覆盖到 `public/gesture_model.onnx`。

### 3. 启动开发服务器

```bash
npm run dev
```

打开浏览器访问 [http://localhost:3000](http://localhost:3000) 即可使用。

## 📂 项目结构

- `app/`: Next.js 页面路由 (`page.tsx`) 与布局。
- `components/`: UI 组件。
  - `webcam-panel.tsx`: 处理摄像头流、关键点绘制和实时预测循环。
  - `file-upload-panel.tsx`: 处理静态图片上传与预览。
  - `results-panel.tsx` & `confidence-panel.tsx`: 结果显示组件。
- `lib/`: 核心逻辑。
  - `gesture-service.ts`: 单例服务，封装了 MediaPipe 初始化、特征提取、ONNX 推理及结果平滑 (Stabilizer) 逻辑。
- `mlp/`: Python 机器学习相关代码。
  - `convert_to_onnx.py`: 模型导出工具，包含针对 Pipeline 组件的特殊配置。
  - `test_model.py`: 本地 Python 模型测试脚本。

## 💡 使用说明

1.  **切换模式**：点击页面顶部的 Toggle 切换 "Upload File" (上传) 或 "Webcam" (摄像头) 模式。
2.  **摄像头权限**：首次进入 Webcam 模式时，浏览器会请求摄像头权限，请允许。
3.  **手势标准**：请正对摄像头或上传清晰的手部图片。模型支持检测单手手势。
4.  **遇到问题？**
    - 如果画面卡顿或置信度不显示，请按 `F12` 打开控制台查看日志。
    - 确保 `public/` 目录下存在最新的 `gesture_model.onnx` 文件。

---

