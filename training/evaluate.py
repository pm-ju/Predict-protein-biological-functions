"""
Evaluation — compute metrics and generate visualisations for trained models.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    multilabel_confusion_matrix,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import log


def evaluate_model(
    results: dict,
    output_dir: Path = None,
    tag: str = "",
) -> dict:
    """
    Evaluate a trained model's predictions and save metrics + plots.
    
    Args:
        results: dict from train_model() containing preds, probs, labels, masks, history.
        output_dir: directory to save metrics and plots.
        tag: optional string to prefix output files.
    
    Returns:
        dict with computed metrics.
    """
    output_dir = output_dir or config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = results["model"]
    preds = results["preds"]
    probs = results["probs"]
    labels = results["labels"]
    test_mask = results["test_mask"]

    prefix = f"{tag}_{model_name}" if tag else model_name

    # ── Test set metrics ──────────────────────────────────────────────────
    y_true = labels[test_mask]
    y_pred = preds[test_mask]
    y_prob = probs[test_mask]

    metrics = {}

    # Subset accuracy (exact match ratio)
    metrics["subset_accuracy"] = float(accuracy_score(y_true, y_pred))

    # F1 scores
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Precision & Recall
    metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

    # ROC-AUC (per-label, then averaged)
    try:
        # Only compute for labels that appear in both true and pred
        valid_cols = np.where(y_true.sum(axis=0) > 0)[0]
        if len(valid_cols) > 0:
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true[:, valid_cols], y_prob[:, valid_cols], average="macro")
            )
        else:
            metrics["roc_auc_macro"] = 0.0
    except ValueError:
        metrics["roc_auc_macro"] = 0.0

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics["per_class_f1"] = {
        config.TARGET_GO_TERMS[i]: round(float(f), 4)
        for i, f in enumerate(per_class_f1) if i < len(config.TARGET_GO_TERMS)
    }

    log.info(f"{'='*60}")
    log.info(f"Results for {model_name.upper()}{f' ({tag})' if tag else ''}:")
    log.info(f"  Subset Accuracy : {metrics['subset_accuracy']:.4f}")
    log.info(f"  F1 Micro        : {metrics['f1_micro']:.4f}")
    log.info(f"  F1 Macro        : {metrics['f1_macro']:.4f}")
    log.info(f"  ROC-AUC Macro   : {metrics['roc_auc_macro']:.4f}")
    log.info(f"{'='*60}")

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics_file = output_dir / f"{prefix}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics to {metrics_file}")

    # ── Plot training curves ──────────────────────────────────────────────
    history = results.get("history", {})
    if history.get("train_loss"):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(history["train_loss"], label="Train Loss", color="#4A90D9")
        ax.plot(history["val_loss"], label="Val Loss", color="#E74C3C")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Cross-Entropy Loss")
        ax.set_title(f"{model_name.upper()} Training Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        curve_file = output_dir / f"{prefix}_training_curve.png"
        plt.savefig(curve_file, dpi=150)
        plt.close()
        log.info(f"Saved training curve to {curve_file}")

    # ── Per-class F1 bar chart ────────────────────────────────────────────
    if metrics.get("per_class_f1"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        terms = list(metrics["per_class_f1"].keys())
        f1s = list(metrics["per_class_f1"].values())
        colors = sns.color_palette("viridis", len(terms))
        ax.barh(terms, f1s, color=colors)
        ax.set_xlabel("F1 Score")
        ax.set_title(f"{model_name.upper()} Per-Class F1 Score")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        f1_file = output_dir / f"{prefix}_per_class_f1.png"
        plt.savefig(f1_file, dpi=150)
        plt.close()
        log.info(f"Saved per-class F1 chart to {f1_file}")

    return metrics


def compare_models(
    all_metrics: dict[str, dict],
    output_dir: Path = None,
    title: str = "Model Comparison",
) -> None:
    """
    Create a comparison bar chart across multiple models/experiments.
    
    Args:
        all_metrics: { "model_name": metrics_dict, ... }
    """
    output_dir = output_dir or config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = ["f1_micro", "f1_macro", "roc_auc_macro", "subset_accuracy"]
    model_names = list(all_metrics.keys())

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]

    colors = sns.color_palette("Set2", len(model_names))

    for ax, metric in zip(axes, metric_names):
        values = [all_metrics[m].get(metric, 0) for m in model_names]
        bars = ax.bar(model_names, values, color=colors)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    comp_file = output_dir / f"comparison_{title.lower().replace(' ', '_')}.png"
    plt.savefig(comp_file, dpi=150)
    plt.close()
    log.info(f"Saved comparison chart to {comp_file}")
