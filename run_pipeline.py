"""
GNN for Protein Function Prediction — Pipeline Orchestrator
Runs the complete 5-stage pipeline end-to-end.

Usage:
    python run_pipeline.py                      # Full pipeline
    python run_pipeline.py --subset-size 100    # Small demo
    python run_pipeline.py --stage 3            # Start from stage 3
    python run_pipeline.py --ablation           # Run ablation experiments
"""

import argparse
import json
import numpy as np
from pathlib import Path

import config
from utils.helpers import log, set_seed


def run_pipeline(
    subset_size: int = None,
    start_stage: int = 1,
    num_epochs: int = None,
    run_ablations: bool = False,
    identity_threshold: float = None,
):
    """Execute the full pipeline from data collection to evaluation."""

    subset_size = subset_size or config.SUBSET_SIZE
    num_epochs = num_epochs or config.NUM_EPOCHS
    identity_threshold = identity_threshold or config.DEFAULT_IDENTITY_THRESHOLD

    set_seed(config.RANDOM_SEED)

    log.info("=" * 70)
    log.info("  GNN for Protein Function Prediction — Pipeline")
    log.info("=" * 70)
    log.info(f"  Subset size     : {subset_size}")
    log.info(f"  Identity thresh : {identity_threshold}")
    log.info(f"  Model epochs    : {num_epochs}")
    log.info(f"  Device          : {config.DEVICE}")
    log.info(f"  Starting stage  : {start_stage}")
    log.info("=" * 70)

    # ── Stage 1: Data Collection ──────────────────────────────────────────
    if start_stage <= 1:
        log.info("\n▶ STAGE 1 — Data Collection")
        from data.download_swissprot import download_proteins
        proteins_csv, labels_npy = download_proteins(subset_size=subset_size)
    else:
        proteins_csv = config.DATA_DIR / "proteins.csv"
        labels_npy = config.DATA_DIR / "labels.npy"
        log.info("⏭ Skipping Stage 1 (data exists)")

    # ── Stage 2: Similarity Computation ───────────────────────────────────
    if start_stage <= 2:
        log.info("\n▶ STAGE 2 — Similarity Computation")
        from data.download_swissprot import load_sequences
        from similarity.run_mmseqs import compute_similarity

        sequences = load_sequences(proteins_csv)

        # Compute for default threshold
        edges_csv = compute_similarity(sequences, identity_threshold=identity_threshold)

        # Also compute for ablation thresholds if running ablations
        if run_ablations:
            for t in config.IDENTITY_THRESHOLDS:
                if t != identity_threshold:
                    compute_similarity(sequences, identity_threshold=t)
    else:
        edges_csv = config.SIMILARITY_DIR / f"edges_t{int(identity_threshold * 100)}.csv"
        log.info("⏭ Skipping Stage 2 (similarity computed)")

    # ── Stage 3: Graph Construction ───────────────────────────────────────
    if start_stage <= 3:
        log.info("\n▶ STAGE 3 — Graph Construction")
        from graph.build_ssn import build_ssn
        G, graph_features_csv = build_ssn(
            edges_csv=edges_csv,
            proteins_csv=proteins_csv,
            identity_threshold=identity_threshold,
        )
    else:
        log.info("⏭ Skipping Stage 3 (graph built)")

    # ── Stage 4: Node Feature Generation ──────────────────────────────────
    if start_stage <= 4:
        log.info("\n▶ STAGE 4 — Node Feature Generation")

        # 4a: ESM2 embeddings
        from features.esm2_embeddings import generate_esm2_embeddings
        esm2_embeddings, accessions = generate_esm2_embeddings()

        # 4b: Handcrafted features (for ablation)
        from features.handcrafted_features import generate_handcrafted_features
        handcrafted_features, _ = generate_handcrafted_features()
    else:
        esm2_embeddings = np.load(config.FEATURES_DIR / "esm2_embeddings.npy")
        handcrafted_features = np.load(config.FEATURES_DIR / "handcrafted_features.npy")
        import csv
        accessions = []
        with open(config.FEATURES_DIR / "protein_order.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                accessions.append(row[0])
        log.info("⏭ Skipping Stage 4 (features generated)")

    # ── Stage 5: Training & Evaluation ────────────────────────────────────
    log.info("\n▶ STAGE 5 — Training & Evaluation")
    labels = np.load(labels_npy)

    from training.train import train_model
    from training.evaluate import evaluate_model

    # Train default GAT model
    results = train_model(
        esm2_embeddings, labels, accessions, edges_csv,
        model_name="gat",
        identity_threshold=identity_threshold,
        num_epochs=num_epochs,
    )
    metrics = evaluate_model(results)

    log.info(f"\n✅ Pipeline complete! F1-micro={metrics['f1_micro']:.4f}, "
             f"ROC-AUC={metrics['roc_auc_macro']:.4f}")

    # ── Ablation experiments (optional) ───────────────────────────────────
    if run_ablations:
        log.info("\n▶ ABLATION EXPERIMENTS")
        from experiments.ablation import run_all_ablations
        run_all_ablations(num_epochs=num_epochs)

    log.info("\n🎉 All done! Results saved to: " + str(config.RESULTS_DIR))


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN Protein Function Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                        # Full pipeline with defaults
  python run_pipeline.py --subset-size 100      # Small demo (100 proteins)
  python run_pipeline.py --epochs 50            # Quick training
  python run_pipeline.py --stage 5              # Resume from training
  python run_pipeline.py --ablation --epochs 50 # All experiments
        """,
    )
    parser.add_argument("--subset-size", type=int, default=config.SUBSET_SIZE,
                        help=f"Number of proteins to download (default: {config.SUBSET_SIZE})")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Start from this pipeline stage (default: 1)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help=f"Training epochs (default: {config.NUM_EPOCHS})")
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_IDENTITY_THRESHOLD,
                        help=f"Identity threshold (default: {config.DEFAULT_IDENTITY_THRESHOLD})")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation experiments after main pipeline")

    args = parser.parse_args()

    run_pipeline(
        subset_size=args.subset_size,
        start_stage=args.stage,
        num_epochs=args.epochs,
        run_ablations=args.ablation,
        identity_threshold=args.threshold,
    )
