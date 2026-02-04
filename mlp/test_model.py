import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import sys
import time
import pickle
from collections import deque

class GestureStabilizer:
    def __init__(self, window_size=5, threshold=0.7):
        """
        :param window_size: 平滑窗口大小（帧数），建议 5-10
        :param threshold: 确信度阈值，低于此值视为「未知/不稳定」
        """
        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=window_size)
    
    def update(self, label, prob):
        """
        更新一帧的预测结果，返回平滑后的结果
        """
        self.history.append((label, prob))
        if len(self.history) < self.window_size:
            return label, prob
        
        # 统计窗口内各类别的出现次数
        labels = [h[0] for h in self.history]
        most_common_label = max(set(labels), key=labels.count)
        
        # 计算该主要类别的平均置信度
        same_label_probs = [h[1] for h in self.history if h[0] == most_common_label]
        avg_prob = np.mean(same_label_probs) if same_label_probs else 0.0
        
        # 只有当该标签在窗口内占比 > 60% 且置信度达标时，才确认切换
        if labels.count(most_common_label) >= (self.window_size * 0.6) and avg_prob > self.threshold:
            return most_common_label, avg_prob
        else:
            return "Unstable", avg_prob


# === 基础配置：所有文件都在当前脚本目录下 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gesture_svm_model_1to6.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

print(f"工作目录: {BASE_DIR}")

# === 1. 核心特征提取逻辑 ===
def build_gesture_features(keypoints):
    """
    将原始坐标特征转为「相对距离+角度特征」（23个特征）
    input: 原始坐标特征 (n_samples, 42/63) 或 单个样本
    output: 构建后的特征 (n_samples, 23)
    """
    # 确保输入是 numpy 数组
    keypoints = np.array(keypoints)
    
    # 如果是单个样本，转换为 batch 模式
    if keypoints.ndim == 1:
        keypoints = keypoints.reshape(1, -1)
    
    n_samples = keypoints.shape[0]
    
    # 自动判断 2D 还是 3D (42 -> 2D, 63 -> 3D)
    if keypoints.shape[1] == 42:
        kp_dim = 2
    elif keypoints.shape[1] == 63:
        kp_dim = 3
    else:
        # 如果不是标准形状，可能是传入了 (N, 21, 2) 或 (N, 21, 3)
        kp_dim = keypoints.shape[-1]
        
    kp = keypoints.reshape(n_samples, 21, kp_dim)  # (N, 21, 2/3)
    
    features = []
    for i in range(n_samples):
        single_kp = kp[i]
        f = []
        
        # 1. 基准长度：腕部(0)到中指尖(12)的距离
        base_len = np.linalg.norm(single_kp[0] - single_kp[12])
        base_len = max(base_len, 1e-6)
        
        # 2. 核心相对距离 (15个特征)
        distance_pairs = [
            (4,8), (8,12), (12,16), (16,20), (4,20),  # 指尖间距离
            (0,4), (0,8), (0,12), (0,16), (0,20),     # 腕部到各指尖距离
            (8,5), (5,6), (6,7), (12,9), (20,17)      # 指节间距离
        ]
        for (p1, p2) in distance_pairs:
            dist = np.linalg.norm(single_kp[p1] - single_kp[p2])
            f.append(dist / base_len)
        
        # 3. 关键角度 (8个特征)
        def calc_angle(a, b, c):
            ba = a - b
            bc = c - b
            numerator = np.dot(ba, bc)
            denominator = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
            cos_ang = numerator / denominator
            cos_ang = np.clip(cos_ang, -1, 1)
            return np.arccos(cos_ang) * 180 / np.pi
        
        angle_triples = [
            (4,0,8), (8,0,12), (12,0,16), (16,0,20),
            (0,5,8), (0,9,12), (0,13,16), (0,17,20)
        ]
        for (a, b, c) in angle_triples:
            f.append(calc_angle(single_kp[a], single_kp[b], single_kp[c]))
        
        features.append(f)

    return np.array(features, dtype=np.float64)

# === 2. 加载模型与Scaler ===

# 检查模型是否存在
if not os.path.exists(MODEL_PATH):
    print(f"❌ 错误：找不到模型文件 {MODEL_PATH}")
    sys.exit(1)

model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ 模型加载成功 (pickle)")
except Exception as e:
    print(f"Pickle加载失败，尝试Joblib: {e}")
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ 模型加载成功 (joblib)")
    except Exception as e2:
        print(f"❌ 模型加载失败: {e2}")
        sys.exit(1)

# 检查Scaler是否存在
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler 加载成功: {SCALER_PATH}")
    except Exception as e:
        print(f"⚠️ Scaler 加载失败: {e}")
else:
    print("⚠️ Warning: scaler.pkl 未找到，预测可能不准确！")


# === 3. MediaPipe 初始化 ===
# 初始化平滑器
stabilizer = GestureStabilizer(window_size=8, threshold=0.7)

if not os.path.exists(TASK_PATH):
    print(f"❌ 错误：找不到 MediaPipe 任务文件 {TASK_PATH}")
    print("请确保 hand_landmarker.task 文件位于 mlp 文件夹中")
    sys.exit(1)

print(f"✅ 使用 MediaPipe 模型: {TASK_PATH}")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

try:
    detector = HandLandmarker.create_from_options(options)
except RuntimeError as e:
    print(f"❌ MediaPipe 初始化失败: {e}")
    print("提示: 这通常是由于 hand_landmarker.task 文件损坏或路径不正确导致的。")
    sys.exit(1)

# 定义连接线 (用于绘制)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# === 4. 主循环 ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
timestamp = 0

print("\n📷 开始视频流识别... 按 'ESC' 退出")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # 模拟时间戳 (ms)
        timestamp += 33 
        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # 绘制关键点
                for lm in hand_landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # 绘制连线
                for (start, end) in HAND_CONNECTIONS:
                    p1 = hand_landmarks[start]
                    p2 = hand_landmarks[end]
                    cv2.line(frame, 
                             (int(p1.x * width), int(p1.y * height)),
                             (int(p2.x * width), int(p2.y * height)),
                             (255, 0, 0), 2)
                
                # === 特征构建与预测 ===
                # 提取 3D 坐标
                raw_landmarks = []
                for lm in hand_landmarks:
                    raw_landmarks.extend([lm.x, lm.y, lm.z]) 
                
                try:
                    # 1. 构建特征
                    features_23 = build_gesture_features(np.array(raw_landmarks)) 
                    
                    # 2. 归一化 (如果存在 scaler)
                    if scaler:
                        if features_23.ndim == 1:
                            features_23 = features_23.reshape(1, -1)
                        features_23 = scaler.transform(features_23)

                    # 3. 预测
                    pred_class = model.predict(features_23)[0]
                    prob = 1.0 # 默认置信度

                    # 获取置信度
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(features_23)[0]
                        prob = np.max(probs)
                    
                    # === 4. 平滑处理 ===
                    stable_class, stable_prob = stabilizer.update(pred_class, prob)
                    
                    # 显示结果 (原始结果用小字，稳定结果用大字)
                    cv2.putText(frame, f"Raw: {pred_class} ({prob:.2f})", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

                    if stable_class == "Unstable":
                        color = (0, 0, 150) # 暗红色表示不稳定
                        label_text = "Stabilizing..."
                    else:
                        color = (0, 255, 0) # 绿色表示稳定
                        label_text = f"Gesture: {stable_class}"
                    
                    cv2.putText(frame, label_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.5, color, 3)
                    cv2.putText(frame, f"Conf: {stable_prob:.2f}", (50, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                except Exception as e:
                    print(f"预测过程出错: {e}")

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break

except KeyboardInterrupt:
    print("用户中断")
finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
