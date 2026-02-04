import sys
import os

# 全局异常捕获，防止窗口闪退
def exception_hook(exctype, value, traceback):
    import traceback as tb
    print("\n❌ 发生严重错误 (Critical Error):")
    tb.print_exception(exctype, value, traceback)
    print("\nPlease check the error message above.")
    input("Press Enter to exit... (按回车键退出)")
    sys.exit(1)

sys.excepthook = exception_hook

print("🚀 脚本启动 (Script Started)...", flush=True)

try:
    print("📦 正在导入基础库 (Importing basic libraries)...", flush=True)
    import pickle
    import joblib
    import numpy as np
    
    print("📦 正在导入 sklearn...", flush=True)
    from sklearn.pipeline import Pipeline
    import sklearn
    print(f"  -> Current sklearn version: {sklearn.__version__}", flush=True)

    print("📦 正在导入 ONNX 相关库 (skl2onnx, onnx)...", flush=True)
    
    print("  -> Attempting to import onnx...", flush=True)
    import onnx
    print("  -> onnx imported successfully.", flush=True)

    print("  -> Attempting to import skl2onnx...", flush=True)
    import skl2onnx
    print("  -> skl2onnx imported successfully.", flush=True)
    
    print("  -> Attempting to import convert_sklearn...", flush=True)
    from skl2onnx import convert_sklearn
    print("  -> convert_sklearn imported successfully.", flush=True)

    from skl2onnx.common.data_types import FloatTensorType
    print("✅ 所有库导入成功 (All imports successful).\n", flush=True)

except ImportError as e:
    print(f"\n❌ 缺少必要库 (Import Error): {e}")
    print("请尝试运行以下命令安装 (Try installing dependencies):")
    print("pip install numpy scikit-learn onnx skl2onnx joblib")
    input("Press Enter to exit... (按回车键退出)")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ 初始化失败 (Initialization Failed): {e}")
    input("Press Enter to exit... (按回车键退出)")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gesture_svm_model_1to6.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ONNX_OUTPUT_PATH = os.path.join(BASE_DIR, "gesture_model.onnx")

def load_with_fallback(path, name):
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        input("Press Enter to exit...")
        sys.exit(1)
        
    obj = None
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"✅ {name} 加载成功 (pickle)")
    except Exception as e:
        print(f"⚠️ Pickle 加载 {name} 失败: {e}，尝试 Joblib...")
        try:
            obj = joblib.load(path)
            print(f"✅ {name} 加载成功 (joblib)")
        except Exception as e2:
            print(f"❌ {name} 加载失败: {e2}")
            input("Press Enter to manually exit...")
            sys.exit(1)
    return obj

def main():
    print("🔄 开始加载模型和 Scaler...")
    
    scaler = load_with_fallback(SCALER_PATH, "Scaler")
    model = load_with_fallback(MODEL_PATH, "SVM Model")

    print(f"Scaler 类型: {type(scaler)}")
    print(f"Model 类型: {type(model)}")

    # 创建 Pipeline
    # 这样 ONNX 模型会包含：输入 -> Scaler -> Model -> 输出
    # 前端只需要传入原始特征，无需手动归一化
    pipeline = Pipeline([
        ('scaler', scaler),
        ('svm', model)
    ])
    
    print("🔄 正在转换为 ONNX 格式...")
    
    # 定义输入类型：23维浮点向量 (根据 test_model.py 的 build_gesture_features)
    # None 表示 Batch Size 维度未知 (支持任意 batch)
    initial_type = [('float_input', FloatTensorType([None, 23]))]
    
    options = {}
    
    # 针对 Pipeline 中的每一步 (只对支持 zipmap 的分类器设置)
    for step_name, step_obj in pipeline.steps:
        # 只针对 SVC 或分类器设置 zipmap: False
        # StandardScaler 不支持这个选项，所以之前报错了
        class_name = type(step_obj).__name__
        if "SVC" in class_name or "Classifier" in class_name:
            options[type(step_obj)] = {'zipmap': False}
    
    print(f"  -> Generated options: {options}", flush=True)

    try:
        print("  -> Converting with zipmap=False...", flush=True)
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type, options=options)
        
        # 保存模型
        with open(ONNX_OUTPUT_PATH, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        print(f"\n🎉 转换成功！ONNX 模型已保存至:\n{ONNX_OUTPUT_PATH}")
        print("\n下一步：")
        print("1. 将此 .onnx 文件移动到前端项目的 public 目录。")
        print("2. 在前端使用 onnxruntime-web 加载并运行。")
        
    except Exception as e:
        print(f"\n❌ ONNX 转换失败: {e}")
        print("这可能是因为 sklearn 版本不兼容。")
        print("如果是版本问题，请尝试仅导出 SVM，并在 JS 中手动实现 Scaler。")
    
    input("Press Enter to complete... (按回车键结束)")

if __name__ == "__main__":
    main()
