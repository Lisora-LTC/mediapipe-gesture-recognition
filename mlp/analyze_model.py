
import pickle
import joblib
import json
import numpy as np
import os

MODEL_PATH = "gesture_svm_model_1to6.pkl"
SCALER_PATH = "scaler.pkl"

def analyze():
    print("Loading model...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except:
        model = joblib.load(MODEL_PATH)
    
    print(f"Model Type: {type(model)}")
    print(f"Model Params: {model.get_params()}")
    
    # Check if linear
    if hasattr(model, 'coef_'):
        print("Model has coef_ (Linear kernel possible)")
        print(f"Coef shape: {model.coef_.shape}")
        print(f"Intercept shape: {model.intercept_.shape}")
        
        # Save weights for JS if linear
        weights = {
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "classes": model.classes_.tolist()
        }
        with open("model_weights.json", "w") as f:
            json.dump(weights, f)
        print("Saved model_weights.json")
    else:
        print("Model does not have coef_ (Non-linear kernel?)")

    print("\nLoading scaler...")
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler Type: {type(scaler)}")
    
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        scaler_params = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        }
        with open("scaler_params.json", "w") as f:
            json.dump(scaler_params, f)
        print("Saved scaler_params.json")

if __name__ == "__main__":
    analyze()
