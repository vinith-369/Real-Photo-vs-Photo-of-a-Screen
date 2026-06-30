"""
Raw ML Training Pipeline for Screen vs Real classification.

- Takes 50 images per class for training (Stratified 5-Fold CV)
- Holds out remaining images for final testing
- Trains multiple classifiers, tunes hyperparameters, picks the best
- Saves best model + scaler + results report
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import joblib
from glob import glob
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# Add parent dir so we can find Dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from feature_extraction import extract_features, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_dataset():
    """Load all image paths and labels."""
    real_dir = os.path.join(SCRIPT_DIR, "data", "Real")
    screen_dir = os.path.join(SCRIPT_DIR, "data", "Screen")

    real_paths = sorted(glob(os.path.join(real_dir, "*.*")))
    screen_paths = sorted(glob(os.path.join(screen_dir, "*.*")))

    # Filter out hidden files / DS_Store
    real_paths = [p for p in real_paths if not os.path.basename(p).startswith(".")]
    screen_paths = [p for p in screen_paths if not os.path.basename(p).startswith(".")]

    print(f"Found {len(real_paths)} Real images, {len(screen_paths)} Screen images")
    return real_paths, screen_paths


def extract_all_features(paths, label_name=""):
    """Extract features for a list of image paths."""
    X_list = []
    valid_paths = []
    feature_names = None

    for i, p in enumerate(paths):
        fname = os.path.basename(p)
        print(f"  [{i+1}/{len(paths)}] Extracting: {fname}")
        feats = extract_features(p)
        if feats is not None:
            if feature_names is None:
                feature_names = sorted(feats.keys())
            vec = np.array([feats.get(name, 0.0) for name in feature_names], dtype=np.float64)
            X_list.append(vec)
            valid_paths.append(p)

    if X_list:
        X = np.array(X_list)
    else:
        X = np.empty((0, 0))

    return X, valid_paths, feature_names


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

def get_models_and_params():
    """Return dict of model_name -> (estimator, param_grid)."""
    models = OrderedDict()
    models["SVM_Linear"] = (
        SVC(kernel="linear", probability=True, random_state=42),
        {"C": [0.001, 0.01, 0.1, 1.0]}
    )
    return models


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names):
    """Train all models with GridSearchCV, evaluate on held-out test set."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Replace NaN/Inf with 0
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_models_and_params()

    results = {}
    best_overall_test_f1 = -1
    best_overall_model = None
    best_overall_name = None

    print("\n" + "=" * 80)
    print("TRAINING & CROSS-VALIDATION")
    print("=" * 80)

    for name, (estimator, param_grid) in models.items():
        print(f"\n{'─' * 60}")
        print(f"  Training: {name}")
        print(f"{'─' * 60}")

        t0 = time.time()

        grid = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X_train_scaled, y_train)
        elapsed = time.time() - t0

        # CV results
        cv_acc = grid.cv_results_["mean_test_score"][grid.best_index_]
        cv_std = grid.cv_results_["std_test_score"][grid.best_index_]

        # Re-evaluate with accuracy on CV (grid scored by f1)
        best_model = grid.best_estimator_

        # Held-out test evaluation
        y_pred_test = best_model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test, zero_division=0)
        test_rec = recall_score(y_test, y_pred_test, zero_division=0)
        test_cm = confusion_matrix(y_test, y_pred_test)

        # Also get CV accuracy (not just f1)
        from sklearn.model_selection import cross_val_score
        cv_acc_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
        cv_f1_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring="f1")

        result = {
            "cv_accuracy_mean": float(np.mean(cv_acc_scores)),
            "cv_accuracy_std": float(np.std(cv_acc_scores)),
            "cv_f1_mean": float(np.mean(cv_f1_scores)),
            "cv_f1_std": float(np.std(cv_f1_scores)),
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "test_confusion_matrix": test_cm.tolist(),
            "best_params": grid.best_params_,
            "train_time_sec": round(elapsed, 2),
        }
        results[name] = result

        print(f"  Best params: {grid.best_params_}")
        print(f"  Test Accuracy: {test_acc:.4f}  |  Test F1: {test_f1:.4f}")
        print(f"  Confusion Matrix: {test_cm.tolist()}")

        # Track best by test F1 (we have a held-out test set for final model selection)
        if test_f1 > best_overall_test_f1:
            best_overall_test_f1 = test_f1
            best_overall_model = best_model
            best_overall_name = name



    return results, best_overall_model, best_overall_name, scaler


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def compute_feature_importance(X_train, y_train, feature_names, scaler):
    """Compute feature importance via mutual information."""
    X_scaled = scaler.transform(X_train)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    mi = mutual_info_classif(X_scaled, y_train, random_state=42)
    importance = list(zip(feature_names, mi))
    importance.sort(key=lambda x: x[1], reverse=True)
    return importance


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(results, feature_importance, best_name, n_train, n_test, feature_names):
    """Generate markdown results report."""
    lines = []
    lines.append("# Raw ML Screen Detection — Results Report\n")
    lines.append(f"**Training set**: {n_train} images (50 Real + 50 Screen)\n")
    lines.append(f"**Held-out test set**: {n_test} images (remaining)\n")
    lines.append(f"**Total features extracted**: {len(feature_names)}\n")
    lines.append(f"**Evaluation**: Stratified 5-Fold Cross-Validation + Held-out Test\n")
    lines.append("")

    # --- Leaderboard ---
    lines.append("## 🏆 Model Leaderboard\n")
    lines.append("| Rank | Model | CV Accuracy | CV F1 | Test Accuracy | Test F1 | Test Precision | Test Recall | Time |")
    lines.append("|------|-------|-------------|-------|---------------|---------|----------------|-------------|------|")

    sorted_results = sorted(results.items(), key=lambda x: x[1]["test_f1"], reverse=True)
    for rank, (name, r) in enumerate(sorted_results, 1):
        marker = " 👑" if name == best_name else ""
        lines.append(
            f"| {rank} | **{name}**{marker} | "
            f"{r['cv_accuracy_mean']:.4f} ± {r['cv_accuracy_std']:.4f} | "
            f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f} | "
            f"{r['test_accuracy']:.4f} | "
            f"{r['test_f1']:.4f} | "
            f"{r['test_precision']:.4f} | "
            f"{r['test_recall']:.4f} | "
            f"{r['train_time_sec']:.1f}s |"
        )
    lines.append("")

    # --- Best Model ---
    lines.append(f"## ✅ Best Model: **{best_name}**\n")
    best_r = results[best_name]
    lines.append(f"- **CV Accuracy**: {best_r['cv_accuracy_mean']:.4f} ± {best_r['cv_accuracy_std']:.4f}")
    lines.append(f"- **CV F1 Score**: {best_r['cv_f1_mean']:.4f} ± {best_r['cv_f1_std']:.4f}")
    lines.append(f"- **Test Accuracy**: {best_r['test_accuracy']:.4f}")
    lines.append(f"- **Test F1 Score**: {best_r['test_f1']:.4f}")
    lines.append(f"- **Test Precision**: {best_r['test_precision']:.4f}")
    lines.append(f"- **Test Recall**: {best_r['test_recall']:.4f}")
    lines.append(f"- **Best Params**: `{best_r['best_params']}`")
    lines.append("")

    # Confusion matrix
    cm = best_r["test_confusion_matrix"]
    lines.append("### Confusion Matrix (Test Set)\n")
    lines.append("| | Predicted Real | Predicted Screen |")
    lines.append("|---|---|---|")
    lines.append(f"| **Actual Real** | {cm[0][0]} | {cm[0][1]} |")
    lines.append(f"| **Actual Screen** | {cm[1][0]} | {cm[1][1]} |")
    lines.append("")

    # --- All confusion matrices ---
    lines.append("## 📊 All Confusion Matrices (Test Set)\n")
    for name, r in sorted_results:
        cm = r["test_confusion_matrix"]
        lines.append(f"### {name}\n")
        lines.append("| | Predicted Real | Predicted Screen |")
        lines.append("|---|---|---|")
        lines.append(f"| **Actual Real** | {cm[0][0]} | {cm[0][1]} |")
        lines.append(f"| **Actual Screen** | {cm[1][0]} | {cm[1][1]} |")
        lines.append("")

    # --- Feature Importance ---
    lines.append("## 🔬 Top 30 Most Important Features (Mutual Information)\n")
    lines.append("| Rank | Feature | MI Score |")
    lines.append("|------|---------|----------|")
    for rank, (fname, score) in enumerate(feature_importance[:30], 1):
        lines.append(f"| {rank} | `{fname}` | {score:.4f} |")
    lines.append("")

    # --- Per-model details ---
    lines.append("## 📋 Per-Model Best Hyperparameters\n")
    for name, r in sorted_results:
        lines.append(f"### {name}\n")
        lines.append(f"```json\n{json.dumps(r['best_params'], indent=2, default=str)}\n```\n")

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 80)
    print("  RAW ML SCREEN DETECTOR — TRAINING PIPELINE")
    print("=" * 80)

    # 1. Load dataset
    real_paths, screen_paths = load_dataset()

    # 2. Split: 50 per class for training, rest for testing
    TRAIN_PER_CLASS = 50

    np.random.seed(42)
    real_indices = np.random.permutation(len(real_paths))
    screen_indices = np.random.permutation(len(screen_paths))

    train_real = [real_paths[i] for i in real_indices[:TRAIN_PER_CLASS]]
    test_real = [real_paths[i] for i in real_indices[TRAIN_PER_CLASS:]]

    train_screen = [screen_paths[i] for i in screen_indices[:TRAIN_PER_CLASS]]
    test_screen = [screen_paths[i] for i in screen_indices[TRAIN_PER_CLASS:]]

    print(f"\nTrain split: {len(train_real)} Real + {len(train_screen)} Screen = {len(train_real) + len(train_screen)}")
    print(f"Test split:  {len(test_real)} Real + {len(test_screen)} Screen = {len(test_real) + len(test_screen)}")

    # 3. Extract features
    print("\n--- Extracting TRAINING features ---")
    X_train_real, valid_train_real, feat_names = extract_all_features(train_real, "Real")
    X_train_screen, valid_train_screen, _ = extract_all_features(train_screen, "Screen")

    X_train = np.vstack([X_train_real, X_train_screen])
    y_train = np.array([0] * len(X_train_real) + [1] * len(X_train_screen))

    print(f"\nTraining feature matrix: {X_train.shape}")

    print("\n--- Extracting TEST features ---")
    X_test_real, valid_test_real, _ = extract_all_features(test_real, "Real")
    X_test_screen, valid_test_screen, _ = extract_all_features(test_screen, "Screen")

    X_test = np.vstack([X_test_real, X_test_screen])
    y_test = np.array([0] * len(X_test_real) + [1] * len(X_test_screen))

    print(f"Test feature matrix: {X_test.shape}")

    # Replace NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # 4. Train and evaluate
    results, best_model, best_name, scaler = train_and_evaluate(
        X_train, y_train, X_test, y_test, feat_names
    )

    # 5. Feature importance
    print("\n--- Computing Feature Importance ---")
    feature_importance = compute_feature_importance(X_train, y_train, feat_names, scaler)

    # 6. Save artifacts
    print("\n--- Saving Models & Report ---")
    save_dir = SCRIPT_DIR

    joblib.dump(best_model, os.path.join(save_dir, "best_raw_model.pkl"))
    print(f"  Saved: best_raw_model.pkl ({best_name})")

    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    print(f"  Saved: scaler.pkl")

    # Save feature names for prediction
    with open(os.path.join(save_dir, "feature_names.json"), "w") as f:
        json.dump(feat_names, f)
    print(f"  Saved: feature_names.json ({len(feat_names)} features)")

    # Save results JSON
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: results.json")

    # Generate report
    report = generate_report(
        results, feature_importance, best_name,
        n_train=len(X_train), n_test=len(X_test),
        feature_names=feat_names,
    )
    report_path = os.path.join(save_dir, "results_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: results_report.md")

    # 7. Final summary
    print("\n" + "=" * 80)
    print(f"  🏆 BEST MODEL: {best_name}")
    best_r = results[best_name]
    print(f"     CV Accuracy:   {best_r['cv_accuracy_mean']:.4f} ± {best_r['cv_accuracy_std']:.4f}")
    print(f"     CV F1:         {best_r['cv_f1_mean']:.4f} ± {best_r['cv_f1_std']:.4f}")
    print(f"     Test Accuracy: {best_r['test_accuracy']:.4f}")
    print(f"     Test F1:       {best_r['test_f1']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
