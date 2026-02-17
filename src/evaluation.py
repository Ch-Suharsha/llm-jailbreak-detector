"""
evaluation.py — Generate evaluation metrics, visualizations, and feature importance.

Outputs:
  - Classification report (printed)
  - Confusion matrix heatmap → models/confusion_matrix.png
  - ROC curve + AUC → models/roc_curve.png
  - Feature importance chart → models/feature_importance.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import joblib

from feature_engineering import get_handcrafted_feature_names

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# Plot styling
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGSIZE = (10, 7)


# ---------------------------------------------------------------------------
# Load predictions and metadata
# ---------------------------------------------------------------------------

def load_evaluation_data():
    """Load test predictions and model metadata."""
    pred_path = DATA_DIR / "test_predictions.csv"
    meta_path = MODEL_DIR / "model_metadata.json"

    if not pred_path.exists() or not meta_path.exists():
        print("Predictions not found. Please run model_training.py first.")
        return None, None

    preds = pd.read_csv(pred_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    return preds, metadata


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Jailbreak"],
        yticklabels=["Benign", "Jailbreak"],
        ax=ax,
        annot_kws={"size": 18},
    )
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_title("Confusion Matrix — Jailbreak Detection", fontsize=16, fontweight="bold")

    # Add counts annotation
    tn, fp, fn, tp = cm.ravel()
    stats_text = f"TP={tp}  TN={tn}  FP={fp}  FN={fn}"
    ax.text(0.5, -0.15, stats_text, transform=ax.transAxes, ha="center", fontsize=12, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def plot_roc_curve(y_true, y_proba, save_path: Path):
    """Generate and save ROC curve with AUC score."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")

    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curve — Jailbreak Detection", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved → {save_path}")

    return roc_auc


def plot_precision_recall_curve(y_true, y_proba, save_path: Path):
    """Generate precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(recall, precision, color="#4CAF50", lw=2.5, label=f"PR Curve (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.1, color="#4CAF50")

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title("Precision-Recall Curve — Jailbreak Detection", fontsize=16, fontweight="bold")
    ax.legend(loc="lower left", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PR curve saved → {save_path}")


def plot_feature_importance(model, save_path: Path, top_n: int = 20):
    """
    Extract and plot feature importance from the trained model.
    Shows top N most important features.
    """
    # Get the actual classifier from the pipeline
    clf = model
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)

    # Get feature names
    hc_names = get_handcrafted_feature_names()
    emb_names = [f"emb_{i}" for i in range(384)]
    all_names = hc_names + emb_names

    # Get importance scores
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        print("  Model does not support feature importance extraction.")
        return

    # Create DataFrame and sort
    imp_df = pd.DataFrame({
        "feature": all_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # Separate hand-crafted vs embedding features
    hc_imp = imp_df[imp_df["feature"].isin(hc_names)].head(top_n)
    emb_imp = imp_df[imp_df["feature"].str.startswith("emb_")]

    # Aggregate embedding importance
    emb_total = emb_imp["importance"].sum()
    hc_total = hc_imp["importance"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Top hand-crafted features
    if not hc_imp.empty:
        colors = sns.color_palette("viridis", len(hc_imp))
        axes[0].barh(
            hc_imp["feature"].values[::-1],
            hc_imp["importance"].values[::-1],
            color=colors,
        )
        axes[0].set_title("Top Hand-Crafted Features", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Importance Score", fontsize=12)

    # Right: Feature group contribution
    groups = {
        "Jailbreak Keywords": imp_df[imp_df["feature"] == "jailbreak_keyword_freq"]["importance"].sum(),
        "Instruction Patterns": imp_df[imp_df["feature"] == "instruction_pattern_count"]["importance"].sum(),
        "Imperative Verbs": imp_df[imp_df["feature"] == "imperative_verb_score"]["importance"].sum(),
        "Text Length": imp_df[imp_df["feature"].isin(["char_count", "word_count"])]["importance"].sum(),
        "Special Chars": imp_df[imp_df["feature"].isin(["special_char_ratio", "uppercase_ratio", "numeric_ratio"])]["importance"].sum(),
        "Other Structural": imp_df[imp_df["feature"].isin(["sentence_count", "quote_nesting_depth", "newline_count", "exclamation_count", "avg_word_length"])]["importance"].sum(),
        "Embeddings (384d)": emb_total,
    }

    group_df = pd.DataFrame(list(groups.items()), columns=["group", "importance"])
    group_df = group_df.sort_values("importance", ascending=True)

    colors2 = sns.color_palette("coolwarm", len(group_df))
    axes[1].barh(group_df["group"], group_df["importance"], color=colors2)
    axes[1].set_title("Feature Group Contributions", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Total Importance", fontsize=12)

    plt.suptitle("Feature Importance Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance saved → {save_path}")

    # Print summary
    print("\n  Top 10 most important features:")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.6f}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation():
    """Run the full evaluation pipeline."""
    print("=" * 60)
    print("LLM Jailbreak Detection — Model Evaluation")
    print("=" * 60)

    # Load data
    preds, metadata = load_evaluation_data()
    if preds is None:
        return

    y_true = preds["y_true"].values
    y_pred = preds["y_pred"].values
    y_proba = preds["y_proba"].values

    # Classification report
    print(f"\nModel: {metadata['model_name']}")
    print(f"Best CV F1: {metadata['cv_f1_score']:.4f}")
    print(f"\n{'='*60}")
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=["Benign", "Jailbreak"]))

    # Model comparison summary
    print("Model Comparison Summary:")
    print("-" * 40)
    for name, scores in metadata.get("comparison_results", {}).items():
        print(f"  {name:25s} F1 = {scores['mean_f1']:.4f} (+/- {scores['std_f1']:.4f})")

    # Generate visualizations
    print("\nGenerating visualizations ...")
    plot_confusion_matrix(y_true, y_pred, MODEL_DIR / "confusion_matrix.png")
    roc_auc = plot_roc_curve(y_true, y_proba, MODEL_DIR / "roc_curve.png")
    plot_precision_recall_curve(y_true, y_proba, MODEL_DIR / "precision_recall_curve.png")

    # Feature importance
    model_path = MODEL_DIR / "jailbreak_detector.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        plot_feature_importance(model, MODEL_DIR / "feature_importance.png")

    # Summary
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Precision:  {metadata['test_precision']:.4f}  (target: >0.90)")
    print(f"  Recall:     {metadata['test_recall']:.4f}  (target: >0.85)")
    print(f"  F1-Score:   {metadata['test_f1']:.4f}  (target: >0.87)")
    print(f"  ROC AUC:    {roc_auc:.4f}")
    print(f"  Targets:    {'✅ ALL MET' if metadata.get('targets_met') else '⚠️ NOT ALL MET'}")


if __name__ == "__main__":
    run_evaluation()
