"""
Ablation Study Runner
Orchestrates the four ablation experiments described in Section 5 of the README:
  1. GNN (GAT) vs FFN — isolate graph structure contribution
  2. ESM2 vs Handcrafted features — compare feature extractors
  3. Threshold Sweep — vary SSN identity cutoff
  4. GAT vs GCN — attention vs averaging
"""

import csv
import json
import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import log, set_seed
from training.train import train_model
from training.evaluate import evaluate_model, compare_models


def _load_accessions() -> list[str]:
    """Load protein accession order from features/output."""
    order_file = config.FEATURES_DIR / "protein_order.csv"
    accessions = []
    with open(order_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            accessions.append(row[0])
    return accessions


def ablation_gnn_vs_ffn(
    embeddings: np.ndarray,
    labels: np.ndarray,
    accessions: list[str],
    edges_csv: Path,
    num_epochs: int = None,
) -> dict[str, dict]:
    """
    Experiment 1: GAT vs FFN on the same ESM2 features.
    Isolates the contribution of graph structure.
    """
    log.info("=" * 60)
    log.info("ABLATION 1: GNN (GAT) vs Feed-Forward Network")
    log.info("=" * 60)

    all_metrics = {}
    for model_name in ["gat", "ffn"]:
        results = train_model(
            embeddings, labels, accessions, edges_csv,
            model_name=model_name,
            num_epochs=num_epochs,
            save_name=f"abl1_{model_name}",
        )
        metrics = evaluate_model(results, tag="abl1")
        all_metrics[model_name.upper()] = metrics

    compare_models(all_metrics, title="GNN vs FFN")
    return all_metrics


def ablation_esm2_vs_handcrafted(
    esm2_embeddings: np.ndarray,
    handcrafted_features: np.ndarray,
    labels: np.ndarray,
    accessions: list[str],
    edges_csv: Path,
    num_epochs: int = None,
) -> dict[str, dict]:
    """
    Experiment 2: GAT with ESM2 features vs GAT with handcrafted features.
    Demonstrates the value of the protein language model.
    """
    log.info("=" * 60)
    log.info("ABLATION 2: ESM2 Features vs Handcrafted Features")
    log.info("=" * 60)

    all_metrics = {}

    # ESM2
    results = train_model(
        esm2_embeddings, labels, accessions, edges_csv,
        model_name="gat", num_epochs=num_epochs,
        save_name="abl2_gat_esm2",
    )
    all_metrics["GAT+ESM2"] = evaluate_model(results, tag="abl2_esm2")

    # Handcrafted
    results = train_model(
        handcrafted_features, labels, accessions, edges_csv,
        model_name="gat", num_epochs=num_epochs,
        save_name="abl2_gat_hand",
    )
    all_metrics["GAT+Handcrafted"] = evaluate_model(results, tag="abl2_hand")

    compare_models(all_metrics, title="ESM2 vs Handcrafted Features")
    return all_metrics


def ablation_threshold_sweep(
    embeddings: np.ndarray,
    labels: np.ndarray,
    accessions: list[str],
    thresholds: list[float] = None,
    num_epochs: int = None,
) -> dict[str, dict]:
    """
    Experiment 3: Vary the similarity threshold (30%, 50%, 70%).
    Shows how graph density affects model performance.
    """
    log.info("=" * 60)
    log.info("ABLATION 3: Similarity Threshold Sweep")
    log.info("=" * 60)

    thresholds = thresholds or config.IDENTITY_THRESHOLDS
    all_metrics = {}

    for threshold in thresholds:
        t_pct = int(threshold * 100)
        edges_csv = config.SIMILARITY_DIR / f"edges_t{t_pct}.csv"

        if not edges_csv.exists():
            log.warning(f"Edge file for threshold {threshold} not found at {edges_csv}. Skipping.")
            continue

        results = train_model(
            embeddings, labels, accessions, edges_csv,
            model_name="gat",
            identity_threshold=threshold,
            num_epochs=num_epochs,
            save_name=f"abl3_gat_t{t_pct}",
        )
        all_metrics[f"t={t_pct}%"] = evaluate_model(results, tag=f"abl3_t{t_pct}")

    compare_models(all_metrics, title="Threshold Sweep")
    return all_metrics


def ablation_gat_vs_gcn(
    embeddings: np.ndarray,
    labels: np.ndarray,
    accessions: list[str],
    edges_csv: Path,
    num_epochs: int = None,
) -> dict[str, dict]:
    """
    Experiment 4: GAT vs GCN — attention vs simple averaging.
    Validates whether attention improves performance on SSNs.
    """
    log.info("=" * 60)
    log.info("ABLATION 4: GAT vs GCN")
    log.info("=" * 60)

    all_metrics = {}
    for model_name in ["gat", "gcn"]:
        results = train_model(
            embeddings, labels, accessions, edges_csv,
            model_name=model_name,
            num_epochs=num_epochs,
            save_name=f"abl4_{model_name}",
        )
        all_metrics[model_name.upper()] = evaluate_model(results, tag="abl4")

    compare_models(all_metrics, title="GAT vs GCN")
    return all_metrics


def run_all_ablations(num_epochs: int = None) -> dict:
    """Run all four ablation experiments and produce a summary."""
    set_seed(config.RANDOM_SEED)

    esm2_emb = np.load(config.FEATURES_DIR / "esm2_embeddings.npy")
    hand_feat = np.load(config.FEATURES_DIR / "handcrafted_features.npy")
    labels = np.load(config.DATA_DIR / "labels.npy")
    accessions = _load_accessions()
    edges_csv = config.SIMILARITY_DIR / f"edges_t{int(config.DEFAULT_IDENTITY_THRESHOLD*100)}.csv"

    summary = {}

    summary["gnn_vs_ffn"] = ablation_gnn_vs_ffn(
        esm2_emb, labels, accessions, edges_csv, num_epochs
    )
    summary["esm2_vs_handcrafted"] = ablation_esm2_vs_handcrafted(
        esm2_emb, hand_feat, labels, accessions, edges_csv, num_epochs
    )
    summary["threshold_sweep"] = ablation_threshold_sweep(
        esm2_emb, labels, accessions, num_epochs=num_epochs
    )
    summary["gat_vs_gcn"] = ablation_gat_vs_gcn(
        esm2_emb, labels, accessions, edges_csv, num_epochs
    )

    # Save summary
    summary_file = config.RESULTS_DIR / "ablation_summary.json"

    # Convert to serialisable format
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(summary_file, "w") as f:
        json.dump(_clean(summary), f, indent=2)

    log.info(f"All ablation results saved to {summary_file}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    args = parser.parse_args()
    run_all_ablations(num_epochs=args.epochs)
