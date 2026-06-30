# Raw ML Screen Detection — Results Report

**Training set**: 100 images (50 Real + 50 Screen)

**Held-out test set**: 38 images (remaining)

**Total features extracted**: 110

**Evaluation**: Stratified 5-Fold Cross-Validation + Held-out Test


## 🏆 Model Leaderboard

| Rank | Model | CV Accuracy | CV F1 | Test Accuracy | Test F1 | Test Precision | Test Recall | Time |
|------|-------|-------------|-------|---------------|---------|----------------|-------------|------|
| 1 | **SVM_Linear** 👑 | 0.8700 ± 0.0510 | 0.8733 ± 0.0507 | 0.9211 | 0.9268 | 0.9500 | 0.9048 | 1.6s |

## ✅ Best Model: **SVM_Linear**

- **CV Accuracy**: 0.8700 ± 0.0510
- **CV F1 Score**: 0.8733 ± 0.0507
- **Test Accuracy**: 0.9211
- **Test F1 Score**: 0.9268
- **Test Precision**: 0.9500
- **Test Recall**: 0.9048
- **Best Params**: `{'C': 0.01}`

### Confusion Matrix (Test Set)

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 16 | 1 |
| **Actual Screen** | 2 | 19 |

## 📊 All Confusion Matrices (Test Set)

### SVM_Linear

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 16 | 1 |
| **Actual Screen** | 2 | 19 |

## 🔬 Top 30 Most Important Features (Mutual Information)

| Rank | Feature | MI Score |
|------|---------|----------|
| 1 | `fft_band_2_energy` | 0.2275 |
| 2 | `fft_std` | 0.2041 |
| 3 | `fft_band_3_energy` | 0.2015 |
| 4 | `color_sat_mean` | 0.1967 |
| 5 | `fft_band_1_energy` | 0.1913 |
| 6 | `fft_band_4_energy` | 0.1888 |
| 7 | `fft_band_0_energy` | 0.1870 |
| 8 | `color_hue_mean` | 0.1745 |
| 9 | `hough_line_count` | 0.1603 |
| 10 | `fft_total_energy` | 0.1297 |
| 11 | `color_b_entropy` | 0.1245 |
| 12 | `canny_density_50_100` | 0.1238 |
| 13 | `color_rb_corr` | 0.1185 |
| 14 | `lap_var_k5` | 0.0955 |
| 15 | `color_hue_std` | 0.0951 |
| 16 | `lbp_bin_9` | 0.0894 |
| 17 | `color_gb_corr` | 0.0891 |
| 18 | `sobel_mag_std` | 0.0876 |
| 19 | `tenengrad_var` | 0.0861 |
| 20 | `lap_mean_k7` | 0.0857 |
| 21 | `lbp_bin_28` | 0.0837 |
| 22 | `lbp_bin_18` | 0.0799 |
| 23 | `glare_dynamic_range` | 0.0790 |
| 24 | `lbp_bin_16` | 0.0787 |
| 25 | `hough_avg_length` | 0.0783 |
| 26 | `color_cb_std` | 0.0782 |
| 27 | `fft_mean` | 0.0773 |
| 28 | `lbp_bin_31` | 0.0754 |
| 29 | `brenner_mean` | 0.0726 |
| 30 | `color_cb_mean` | 0.0705 |

## 📋 Per-Model Best Hyperparameters

### SVM_Linear

```json
{
  "C": 0.01
}
```
