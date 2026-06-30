"""
Comprehensive hand-crafted feature extraction for Screen vs Real image classification.

Extracts 50+ features across 7 signal domains:
1. Sharpness / Blur Detection
2. Edge Analysis
3. Frequency Domain (Moiré / Screen Grid)
4. Color & Chrominance
5. Texture (LBP)
6. Glare / Highlight Detection
7. Noise Estimation

Each image is reduced to a single feature vector for classical ML classifiers.
"""

import os
import numpy as np
import cv2
from PIL import Image
from scipy import stats as sp_stats

# Optional: HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_image(path, target_size=512):
    """Load image via PIL (HEIC-safe), resize to target, return RGB numpy array."""
    img = Image.open(path).convert("RGB")
    # Resize largest side to target_size, preserve aspect ratio
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img)


def _safe_stat(arr):
    """Return (mean, std, skew, kurtosis) for a 1-D array, nan-safe."""
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = float(np.mean(arr))
    s = float(np.std(arr))
    sk = float(sp_stats.skew(arr.ravel()))
    ku = float(sp_stats.kurtosis(arr.ravel()))
    return m, s, sk, ku


# ---------------------------------------------------------------------------
# 1. Sharpness / Blur features
# ---------------------------------------------------------------------------

def _sharpness_features(gray):
    """Laplacian variance at multiple scales + Tenengrad + Brenner."""
    feats = {}

    # Laplacian variance at kernel sizes 3, 5, 7
    for k in (3, 5, 7):
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
        feats[f"lap_var_k{k}"] = float(lap.var())
        feats[f"lap_mean_k{k}"] = float(np.abs(lap).mean())

    # Tenengrad (Sobel gradient magnitude)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.sqrt(gx ** 2 + gy ** 2)
    feats["tenengrad_mean"] = float(tenengrad.mean())
    feats["tenengrad_var"] = float(tenengrad.var())

    # Brenner gradient (difference between pixels 2 apart)
    brenner = (gray[2:, :].astype(np.float64) - gray[:-2, :].astype(np.float64)) ** 2
    feats["brenner_mean"] = float(brenner.mean())

    return feats


# ---------------------------------------------------------------------------
# 2. Edge analysis features
# ---------------------------------------------------------------------------

def _edge_features(gray):
    """Canny edge density at multiple thresholds + Sobel stats + HoughLines."""
    feats = {}
    h, w = gray.shape
    total_pixels = h * w

    # Canny at multiple thresholds
    for lo, hi in ((50, 100), (100, 200), (150, 250)):
        edges = cv2.Canny(gray, lo, hi)
        density = float(np.sum(edges > 0) / total_pixels)
        feats[f"canny_density_{lo}_{hi}"] = density

    # Sobel gradient statistics
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    m, s, sk, ku = _safe_stat(grad_mag)
    feats["sobel_mag_mean"] = m
    feats["sobel_mag_std"] = s
    feats["sobel_mag_skew"] = sk
    feats["sobel_mag_kurt"] = ku

    # Gradient direction histogram entropy
    grad_dir = np.arctan2(gy, gx + 1e-10)
    hist_dir, _ = np.histogram(grad_dir, bins=36, range=(-np.pi, np.pi))
    hist_dir = hist_dir / (hist_dir.sum() + 1e-10)
    dir_entropy = float(-np.sum(hist_dir * np.log2(hist_dir + 1e-10)))
    feats["grad_dir_entropy"] = dir_entropy

    # HoughLinesP — line detection (screens often have strong parallel lines)
    edges_for_hough = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges_for_hough, 1, np.pi / 180, threshold=50,
                            minLineLength=30, maxLineGap=10)
    if lines is not None:
        feats["hough_line_count"] = float(len(lines))
        lengths = [np.sqrt((l[0][2] - l[0][0]) ** 2 + (l[0][3] - l[0][1]) ** 2) for l in lines]
        feats["hough_avg_length"] = float(np.mean(lengths))
        # Angle concentration — screens tend to have dominant horizontal/vertical lines
        angles = [np.abs(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0] + 1e-10)) for l in lines]
        angle_hist, _ = np.histogram(angles, bins=18, range=(0, np.pi))
        angle_hist = angle_hist / (angle_hist.sum() + 1e-10)
        feats["hough_angle_entropy"] = float(-np.sum(angle_hist * np.log2(angle_hist + 1e-10)))
    else:
        feats["hough_line_count"] = 0.0
        feats["hough_avg_length"] = 0.0
        feats["hough_angle_entropy"] = 0.0

    return feats


# ---------------------------------------------------------------------------
# 3. Frequency domain features (Moiré / screen grid detection)
# ---------------------------------------------------------------------------

def _frequency_features(gray, rgb):
    """FFT analysis on grayscale + chroma channels."""
    feats = {}
    h, w = gray.shape

    # --- Grayscale FFT ---
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    # Mask out DC component
    cy, cx = h // 2, w // 2
    mask = np.ones_like(mag)
    cv2.circle(mask, (cx, cy), 10, 0, -1)
    mag_masked = mag * mask

    total_energy = float(mag_masked.sum())
    feats["fft_total_energy"] = total_energy
    feats["fft_mean"] = float(mag_masked.mean())
    feats["fft_std"] = float(mag_masked.std())

    # High-frequency energy ratio (outer ring vs inner)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = min(cx, cy)

    # 5 radial bands
    n_bands = 5
    for i in range(n_bands):
        r_inner = max_r * i / n_bands
        r_outer = max_r * (i + 1) / n_bands
        band_mask = ((dist >= r_inner) & (dist < r_outer)).astype(np.float64)
        band_energy = float((mag_masked * band_mask).sum())
        feats[f"fft_band_{i}_energy"] = band_energy / (total_energy + 1e-10)

    # Peak detection — count of extreme values in spectrum
    threshold = np.mean(mag_masked) + 3 * np.std(mag_masked)
    feats["fft_peak_count"] = float(np.sum(mag_masked > threshold))

    # --- Chroma FFT (Cr, Cb channels — key for screen detection) ---
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    for ch_idx, ch_name in [(1, "cr"), (2, "cb")]:
        ch = ycrcb[:, :, ch_idx].astype(np.float64)
        f_ch = np.fft.fft2(ch)
        fshift_ch = np.fft.fftshift(f_ch)
        mag_ch = np.log1p(np.abs(fshift_ch))
        mag_ch_masked = mag_ch * mask

        ch_mean = float(mag_ch_masked.mean())
        ch_std = float(mag_ch_masked.std())
        ch_thresh = ch_mean + 3 * ch_std
        feats[f"fft_{ch_name}_peak_count"] = float(np.sum(mag_ch_masked > ch_thresh))
        feats[f"fft_{ch_name}_energy"] = float(mag_ch_masked.sum())
        feats[f"fft_{ch_name}_std"] = ch_std

    return feats


# ---------------------------------------------------------------------------
# 4. Color & Chrominance features
# ---------------------------------------------------------------------------

def _color_features(rgb):
    """Color statistics across multiple color spaces."""
    feats = {}

    # --- YCrCb ---
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    for ch_idx, ch_name in [(0, "y"), (1, "cr"), (2, "cb")]:
        ch = ycrcb[:, :, ch_idx].astype(np.float64)
        m, s, sk, ku = _safe_stat(ch)
        feats[f"color_{ch_name}_mean"] = m
        feats[f"color_{ch_name}_std"] = s
        feats[f"color_{ch_name}_skew"] = sk
        feats[f"color_{ch_name}_kurt"] = ku

    # --- HSV ---
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    for ch_idx, ch_name in [(0, "hue"), (1, "sat"), (2, "val")]:
        ch = hsv[:, :, ch_idx].astype(np.float64)
        m, s, sk, ku = _safe_stat(ch)
        feats[f"color_{ch_name}_mean"] = m
        feats[f"color_{ch_name}_std"] = s

    # --- Inter-channel correlation ---
    r, g, b = rgb[:, :, 0].ravel().astype(np.float64), rgb[:, :, 1].ravel().astype(np.float64), rgb[:, :, 2].ravel().astype(np.float64)
    feats["color_rg_corr"] = float(np.corrcoef(r, g)[0, 1]) if np.std(r) > 0 and np.std(g) > 0 else 0.0
    feats["color_gb_corr"] = float(np.corrcoef(g, b)[0, 1]) if np.std(g) > 0 and np.std(b) > 0 else 0.0
    feats["color_rb_corr"] = float(np.corrcoef(r, b)[0, 1]) if np.std(r) > 0 and np.std(b) > 0 else 0.0

    # --- Histogram entropy per channel ---
    for ch_idx, ch_name in [(0, "r"), (1, "g"), (2, "b")]:
        hist = cv2.calcHist([rgb], [ch_idx], None, [64], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-10)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        feats[f"color_{ch_name}_entropy"] = entropy

    return feats


# ---------------------------------------------------------------------------
# 5. Texture features (LBP)
# ---------------------------------------------------------------------------

def _lbp_features(gray):
    """Local Binary Pattern histogram features — captures micro-texture."""
    feats = {}

    # Simple LBP implementation (8-neighbour, radius 1)
    h, w = gray.shape
    if h < 3 or w < 3:
        for i in range(10):
            feats[f"lbp_bin_{i}"] = 0.0
        feats["lbp_entropy"] = 0.0
        return feats

    padded = np.pad(gray.astype(np.int16), 1, mode='reflect')
    center = padded[1:-1, 1:-1]

    # 8 neighbors
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    lbp = np.zeros((h, w), dtype=np.uint8)
    for i, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
        lbp += ((neighbor >= center).astype(np.uint8)) << i

    # Uniform LBP histogram (bin into 10 uniform patterns + 1 non-uniform)
    hist, _ = np.histogram(lbp, bins=32, range=(0, 256))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)

    for i, val in enumerate(hist):
        feats[f"lbp_bin_{i}"] = float(val)

    # LBP entropy
    feats["lbp_entropy"] = float(-np.sum(hist * np.log2(hist + 1e-10)))

    return feats


# ---------------------------------------------------------------------------
# 6. Glare / Highlight detection
# ---------------------------------------------------------------------------

def _glare_features(gray, rgb):
    """Detect bright spots and glare patterns typical of screen photos."""
    feats = {}

    # Percentage of near-white pixels
    white_thresh = 240
    white_ratio = float(np.sum(gray >= white_thresh) / gray.size)
    feats["glare_white_ratio"] = white_ratio

    # Percentage of very dark pixels
    dark_ratio = float(np.sum(gray <= 15) / gray.size)
    feats["glare_dark_ratio"] = dark_ratio

    # Bright spot clustering — use morphological operations
    _, bright_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feats["glare_bright_clusters"] = float(len(contours))
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        feats["glare_max_cluster_area"] = float(max(areas))
        feats["glare_total_cluster_area"] = float(sum(areas))
    else:
        feats["glare_max_cluster_area"] = 0.0
        feats["glare_total_cluster_area"] = 0.0

    # Dynamic range
    feats["glare_dynamic_range"] = float(int(gray.max()) - int(gray.min()))

    # Contrast ratio (95th percentile / 5th percentile)
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    feats["glare_contrast_ratio"] = (p95 - p5)

    return feats


# ---------------------------------------------------------------------------
# 7. Noise estimation
# ---------------------------------------------------------------------------

def _noise_features(gray):
    """Estimate sensor noise characteristics."""
    feats = {}

    # Median-based noise estimation (Donoho's robust estimator)
    # sigma = median(|wavelet detail coefficients|) / 0.6745
    # Approximation: use Laplacian residual
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sigma = float(np.median(np.abs(lap)) / 0.6745)
    feats["noise_sigma"] = sigma

    # Local variance map — divide into blocks and compute variance
    block_size = 16
    h, w = gray.shape
    local_vars = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y + block_size, x:x + block_size].astype(np.float64)
            local_vars.append(float(block.var()))

    if local_vars:
        local_vars = np.array(local_vars)
        feats["noise_local_var_mean"] = float(local_vars.mean())
        feats["noise_local_var_std"] = float(local_vars.std())
        feats["noise_local_var_median"] = float(np.median(local_vars))
        # Coefficient of variation of local variances
        feats["noise_local_var_cv"] = float(local_vars.std() / (local_vars.mean() + 1e-10))
    else:
        feats["noise_local_var_mean"] = 0.0
        feats["noise_local_var_std"] = 0.0
        feats["noise_local_var_median"] = 0.0
        feats["noise_local_var_cv"] = 0.0

    return feats


# ---------------------------------------------------------------------------
# 8. GLCM (Gray-Level Co-occurrence Matrix) texture features
# ---------------------------------------------------------------------------

def _glcm_features(gray):
    """Compute GLCM-based texture features: contrast, dissimilarity, homogeneity, energy, correlation."""
    feats = {}

    # Quantize to 16 levels for speed
    quantized = (gray / 16).astype(np.uint8)
    n_levels = 16
    h, w = quantized.shape

    # Build GLCM for distance=1, angle=0 (horizontal)
    glcm = np.zeros((n_levels, n_levels), dtype=np.float64)
    for y in range(h):
        for x in range(w - 1):
            i, j = quantized[y, x], quantized[y, x + 1]
            glcm[i, j] += 1
            glcm[j, i] += 1  # symmetric

    glcm_sum = glcm.sum()
    if glcm_sum > 0:
        glcm = glcm / glcm_sum

    # Compute properties
    I, J = np.meshgrid(range(n_levels), range(n_levels), indexing='ij')
    I = I.astype(np.float64)
    J = J.astype(np.float64)

    feats["glcm_contrast"] = float(np.sum(glcm * (I - J) ** 2))
    feats["glcm_dissimilarity"] = float(np.sum(glcm * np.abs(I - J)))
    feats["glcm_homogeneity"] = float(np.sum(glcm / (1.0 + (I - J) ** 2)))
    feats["glcm_energy"] = float(np.sum(glcm ** 2))
    feats["glcm_entropy"] = float(-np.sum(glcm * np.log2(glcm + 1e-10)))

    # Correlation
    mu_i = np.sum(I * glcm)
    mu_j = np.sum(J * glcm)
    sigma_i = np.sqrt(np.sum(glcm * (I - mu_i) ** 2))
    sigma_j = np.sqrt(np.sum(glcm * (J - mu_j) ** 2))
    if sigma_i > 0 and sigma_j > 0:
        feats["glcm_correlation"] = float(np.sum(glcm * (I - mu_i) * (J - mu_j)) / (sigma_i * sigma_j))
    else:
        feats["glcm_correlation"] = 0.0

    return feats


# ===========================================================================
# Public API
# ===========================================================================

FEATURE_NAMES = None  # Set on first call


def extract_features(image_path):
    """
    Extract all features from a single image.

    Returns:
        dict: feature_name -> float value, or None if image can't be loaded.
    """
    try:
        rgb = _load_image(image_path, target_size=512)
    except Exception as e:
        print(f"  [WARN] Could not load {image_path}: {e}")
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    all_feats = {}
    all_feats.update(_sharpness_features(gray))
    all_feats.update(_edge_features(gray))
    all_feats.update(_frequency_features(gray, rgb))
    all_feats.update(_color_features(rgb))
    all_feats.update(_lbp_features(gray))
    all_feats.update(_glare_features(gray, rgb))
    all_feats.update(_noise_features(gray))
    all_feats.update(_glcm_features(gray))

    return all_feats


def extract_features_vector(image_path):
    """
    Extract features as a numpy array with consistent ordering.

    Returns:
        (np.ndarray, list[str]) or (None, None)
    """
    feats = extract_features(image_path)
    if feats is None:
        return None, None

    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        FEATURE_NAMES = sorted(feats.keys())

    vec = np.array([feats.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)
    return vec, FEATURE_NAMES
