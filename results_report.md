# Raw ML Screen Detection — Results Report

**Training set**: 100 images (50 Real + 50 Screen)

**Held-out test set**: 38 images (remaining)

**Total features extracted**: 110

**Evaluation**: Stratified 5-Fold Cross-Validation + Held-out Test


## 🏆 Model Leaderboard

| Rank | Model | CV Accuracy | CV F1 | Test Accuracy | Test F1 | Test Precision | Test Recall | Time |
|------|-------|-------------|-------|---------------|---------|----------------|-------------|------|
| 1 | **VotingEnsemble** | 0.8800 ± 0.0678 | 0.8829 ± 0.0666 | 0.9211 | 0.9268 | - | - | - |
| 2 | **SVM_Linear** 👑 | 0.8700 ± 0.0510 | 0.8733 ± 0.0507 | 0.9211 | 0.9268 | 0.9500 | 0.9048 | 0.0s |
| 3 | **StackingEnsemble** | 0.8800 ± 0.0678 | 0.8829 ± 0.0666 | 0.8947 | 0.9000 | - | - | - |
| 4 | **SVM_RBF** | 0.8700 ± 0.0400 | 0.8714 ± 0.0394 | 0.8947 | 0.9091 | 0.8696 | 0.9524 | 0.1s |
| 5 | **LogisticRegression** | 0.8700 ± 0.0510 | 0.8723 ± 0.0492 | 0.8947 | 0.9000 | 0.9474 | 0.8571 | 0.9s |
| 6 | **ExtraTrees** | 0.8800 ± 0.0245 | 0.8805 ± 0.0315 | 0.8684 | 0.8837 | 0.8636 | 0.9048 | 14.7s |
| 7 | **BaggingSVM** | 0.8600 ± 0.0583 | 0.8704 ± 0.0543 | 0.8684 | 0.8837 | 0.8636 | 0.9048 | 1.4s |
| 8 | **RandomForest** | 0.8600 ± 0.0583 | 0.8576 ± 0.0649 | 0.8684 | 0.8837 | 0.8636 | 0.9048 | 19.0s |
| 9 | **GradientBoosting** | 0.8600 ± 0.0735 | 0.8568 ± 0.0769 | 0.8684 | 0.8780 | 0.9000 | 0.8571 | 55.8s |
| 10 | **KNN** | 0.8700 ± 0.0600 | 0.8584 ± 0.0687 | 0.8684 | 0.8780 | 0.9000 | 0.8571 | 0.1s |
| 11 | **HistGradientBoosting** | 0.8200 ± 0.0400 | 0.8137 ± 0.0489 | 0.8421 | 0.8571 | 0.8571 | 0.8571 | 80.1s |
| 12 | **AdaBoost** | 0.8000 ± 0.0316 | 0.7958 ± 0.0337 | 0.8158 | 0.8444 | 0.7917 | 0.9048 | 5.0s |

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

### VotingEnsemble
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 16 | 1 |
| **Actual Screen** | 2 | 19 |

### SVM_RBF
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 14 | 3 |
| **Actual Screen** | 1 | 20 |

### LogisticRegression
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 16 | 1 |
| **Actual Screen** | 3 | 18 |

### StackingEnsemble
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 16 | 1 |
| **Actual Screen** | 3 | 18 |

### ExtraTrees
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 14 | 3 |
| **Actual Screen** | 2 | 19 |

### BaggingSVM
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 14 | 3 |
| **Actual Screen** | 2 | 19 |

### RandomForest
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 14 | 3 |
| **Actual Screen** | 2 | 19 |

### GradientBoosting
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 15 | 2 |
| **Actual Screen** | 3 | 18 |

### KNN
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 15 | 2 |
| **Actual Screen** | 3 | 18 |

### HistGradientBoosting
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 14 | 3 |
| **Actual Screen** | 3 | 18 |

### AdaBoost
| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 12 | 5 |
| **Actual Screen** | 2 | 19 |
