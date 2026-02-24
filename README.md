# GNN for Protein Function Prediction Using Sequence Similarity Networks

Predict protein biological functions by combining **ESM2 protein language model embeddings** with **Graph Attention Networks (GAT)** on **Sequence Similarity Networks (SSNs)**.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (small demo: 100 proteins, 50 epochs)
python run_pipeline.py --subset-size 100 --epochs 50

# Run with ablation experiments
python run_pipeline.py --subset-size 100 --epochs 50 --ablation
```

## Pipeline Overview

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `data/download_swissprot.py` | Download Swiss-Prot proteins with GO annotations |
| 2 | `similarity/run_mmseqs.py` | Compute pairwise sequence similarity (MMseqs2 or k-mer fallback) |
| 3 | `graph/build_ssn.py` | Build SSN graph with NetworkX, compute graph features |
| 4 | `features/esm2_embeddings.py` | Generate 320-dim ESM2 embeddings |
| 4b | `features/handcrafted_features.py` | Compute 26-dim biochemical features |
| 5 | `training/train.py` | Train GAT/GCN/FFN models with PyTorch Geometric |

## Ablation Experiments

1. **GNN vs FFN** — Isolates the contribution of graph structure
2. **ESM2 vs Handcrafted** — Compares protein language model to manual features
3. **Threshold Sweep** — Varies identity cutoff (30%, 50%, 70%)
4. **GAT vs GCN** — Attention vs simple averaging

## Key Tools

- **MMseqs2** — Fast protein similarity computation
- **ESM2** — Meta's protein language model (8M parameter variant)
- **NetworkX** — Graph construction and analysis
- **PyTorch Geometric** — GNN training framework

## Project Structure

```
potein/
├── config.py                # Global configuration
├── run_pipeline.py          # Pipeline orchestrator
├── requirements.txt         # Dependencies
├── data/                    # Stage 1: Data collection
├── similarity/              # Stage 2: Similarity computation
├── graph/                   # Stage 3: SSN construction
├── features/                # Stage 4: Feature generation
├── models/                  # GAT, GCN, FFN architectures
├── training/                # Training loop & evaluation
├── experiments/             # Ablation studies
└── results/                 # Output metrics & plots
```

## Configuration

All parameters are configurable in `config.py`:
- Dataset size, target GO terms
- MMseqs2 identity thresholds
- ESM2 model variant, batch size
- GAT/GCN/FFN architecture params
- Training hyperparameters (lr, epochs, early stopping)
