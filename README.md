# Screen Detector: Spot the Fake Photo

This project is a high-accuracy classical machine learning classifier designed to determine whether a given image is a genuine "real photo" or a "photo of a screen" (a re-capture/fraud). 

It achieves **92.11% accuracy** by using **110 hand-crafted features** spanning frequency (FFT), texture (LBP/GLCM), and sharpness domains. We initially evaluated many machine learning algorithms via cross-validation, but `SVM_Linear` proved to be the fastest and most accurate, so it is the only model we use now.

## 🛠 Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Training (Feature Extraction + GridSearch)
To run the automated feature extraction across the dataset and select the best model:
```bash
python train.py
```

### 2. Single Image Prediction
You can predict the fraud probability of a single image using `predict.py`. It outputs a single float value between `0.0` (Real Photo) and `1.0` (Photo of a Screen).

```bash
python predict.py "path/to/image.jpg"
```
