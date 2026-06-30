"""
Standalone prediction script for the Raw ML Screen Detector.

Usage:
    python raw_ml/predict.py some_image.jpg
    
Prints a single float from 0.0 (Real) to 1.0 (Screen).
"""

import os
import sys
import json
import numpy as np
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from feature_extraction import extract_features


def load_artifacts():
    """Load the saved model, scaler, and feature names."""
    model = joblib.load(os.path.join(SCRIPT_DIR, "best_raw_model.pkl"))
    scaler = joblib.load(os.path.join(SCRIPT_DIR, "scaler.pkl"))
    with open(os.path.join(SCRIPT_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


_model = None
_scaler = None
_feature_names = None


def predict(image_path: str) -> float:
    """
    Predict fraud score for a single image.
    
    Returns:
        float: 0.0 = Real photo, 1.0 = Photo of a screen
    """
    global _model, _scaler, _feature_names
    if _model is None:
        _model, _scaler, _feature_names = load_artifacts()

    feats = extract_features(image_path)
    if feats is None:
        return 0.5  # Uncertain if image can't be loaded

    vec = np.array([feats.get(name, 0.0) for name in _feature_names], dtype=np.float64)
    vec = vec.reshape(1, -1)
    vec = _scaler.transform(vec)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    # Get probability of class 1 (Screen)
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(vec)[0]
        fraud_score = float(proba[1])
    else:
        # Fallback to hard prediction
        pred = _model.predict(vec)[0]
        fraud_score = float(pred)

    return fraud_score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py some_image.jpg")
        sys.exit(1)
    print(f"{predict(sys.argv[1]):.4f}")
