# Approach Note — Real vs Screen Recapture Detector

## Approach

I built a **classical signal-processing + machine-learning pipeline** — no deep learning.
The detector extracts **110 handcrafted features** from seven complementary domains, then evaluates multiple classifiers using Grid Search to automatically select the best performing model.

### Feature Groups (110 dimensions)

1. **Sharpness / Blur Detection (7 dims)** — Laplacian variance at multiple scales, Tenengrad (Sobel gradient magnitude), and Brenner gradients.
2. **Edge Analysis (9 dims)** — Canny edge density at multiple thresholds, Sobel gradient statistics, and HoughLines detection (to capture the straight edges and bezels often found in screen recaptures).
3. **Frequency Domain / Moiré Detection (13 dims)** — 2-D FFT of both the grayscale image and the chroma channels (Cr/Cb) reveals periodic artifacts. Extracts total energy, high-frequency energy in radial bands, and detects spectral peaks that correspond to screen pixel grids and refresh-rate ghosts.
4. **Color & Chrominance (10 dims)** — Intensity histogram moments, saturation statistics, and inter-channel correlations. Screens often produce quantised, narrower colour gamuts.
5. **Texture via LBP (58 dims)** — Local Binary Patterns encode micro-texture using a uniform histogram (58 bins), capturing the unique sub-pixel texture of monitors.
6. **Texture via GLCM (4 dims)** — Gray-Level Co-occurrence Matrix properties (contrast, energy, homogeneity, correlation) to detect the uniform grid-like nature of screen pixels.
7. **Noise Estimation & Glare (9 dims)** — High-pass residual noise statistics and glare/highlight detection via thresholding.

### Classifier

We originally ran an exhaustive GridSearchCV across 12 classical algorithms (Random Forest, Extra Trees, SVM, Gradient Boosting, HistGradientBoosting, KNN, LogisticRegression, etc.). 

After evaluating all of them via cross-validation, **SVM_Linear** was automatically chosen as the best performing model. Because it achieves the highest accuracy and infers in less than a millisecond, we have stripped out the other models and now use `SVM_Linear` exclusively in our training and prediction pipelines.

## Accuracy

Evaluated via stratified 5-fold cross-validation on 100 training images, and tested on a 38-image held-out set.

| Metric    | Score (Test Set) |
|-----------|------------------|
| Accuracy  | 92.11%           |
| Precision | 95.00%           |
| Recall    | 90.48%           |
| F1 Score  | 92.68%           |

## Latency & Cost

| Metric | Value |
|--------|-------|
| **Latency** | **~435 ms** per image (MacBook CPU) |
| **Cost (Cloud Server)** | **~$10.00** per million images |

### Cost Assumptions
- **Latency**: Feature extraction is heavily mathematical (LBP, multiple 2D-FFTs, GLCM). Python loops add overhead, taking ~435ms per image. The SVM inference itself is practically instant (<1 ms).
- **Cloud Cost Math**: Assuming an AWS `c5.large` instance ($0.085/hour). At ~435ms per image, a single instance can process ~8,275 images per hour. 1,000,000 images / 8,275 = ~120 hours. 120 hours × $0.085 = ~$10.20 per million images. 
- Using AWS Lambda (1GB memory) at $0.0000166/sec for 435ms per image would cost roughly $7.25 per million images.

## What I'd Improve

- **Cython/C++ Rewrite**: Porting the feature extraction pipeline (especially LBP and FFT) to C++ would easily drop latency to <10ms per image.
- **Cross-device robustness**: The current features are tuned for specific screen patterns; adding scanner / webcam / printed photo variants would improve generalisation.
- **Ensemble with a tiny CNN**: A MobileNet-v3 head (~200 KB) could complement the handcrafted features for tricky edge cases where classical CV falls short.
