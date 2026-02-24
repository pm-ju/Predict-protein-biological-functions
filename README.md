# GNN for Protein Function Prediction Using Sequence Similarity Networks

Predict protein biological functions by combining **ESM2 protein language model embeddings** with **Graph Neural Networks (GAT / GCN)** on **Sequence Similarity Networks (SSNs)** constructed via **MMseqs2**.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (5000 proteins, 200 epochs, with ablation experiments)
export USE_CUDA=1  # Enable GPU acceleration
python run_pipeline.py --subset-size 5000 --threshold 0.15 --epochs 200 --ablation

# Quick demo (100 proteins, CPU-only)
python run_pipeline.py --subset-size 100 --epochs 50
```

> **Tip:** Install [MMseqs2](https://github.com/soedinglab/MMseqs2) for dramatically faster similarity computation. Without it, the pipeline falls back to a slower Python k-mer method.

---

## Results (5,000 Proteins · MMseqs2 · CUDA)

Evaluated on **5,000 reviewed Swiss-Prot proteins** with 8 GO Molecular Function labels. The SSN was built using MMseqs2 all-vs-all search.

### Graph Statistics
| Metric | Value |
|--------|-------|
| Nodes | 5,000 |
| Edges | 142,206 |
| Density | 0.0114 |
| Avg. Degree | 56.88 |
| Avg. Clustering Coeff. | 0.656 |
| Communities (Louvain) | 1,246 |

### Main Model (GAT · ESM2 · t=0.15)
| Metric | Score |
|--------|-------|
| **F1 Micro** | **0.790** |
| **F1 Macro** | **0.567** |
| **ROC-AUC Macro** | **0.901** |
| Subset Accuracy | 0.619 |

### Ablation 1 — GNN vs Feed-Forward (no graph)
| Model | F1 Micro | F1 Macro | ROC-AUC | Accuracy |
|-------|----------|----------|---------|----------|
| **GAT** | 0.788 | **0.634** | 0.909 | 0.620 |
| FFN | **0.807** | 0.581 | **0.934** | **0.652** |

### Ablation 2 — ESM2 vs Handcrafted Features
| Features | F1 Micro | F1 Macro | ROC-AUC |
|----------|----------|----------|---------|
| **GAT + ESM2** (320-dim) | **0.788** | **0.634** | **0.909** |
| GAT + Handcrafted (26-dim) | 0.628 | 0.307 | 0.811 |

### Ablation 3 — Similarity Threshold Sweep
| Threshold | Edges | F1 Micro | F1 Macro | ROC-AUC |
|-----------|-------|----------|----------|---------|
| **t=30%** | 62,992 | **0.788** | 0.634 | **0.909** |
| t=50% | 10,241 | 0.779 | 0.576 | 0.896 |
| t=70% | 4,770 | 0.769 | **0.646** | 0.895 |

### Ablation 4 — GAT vs GCN
| Model | F1 Micro | F1 Macro | ROC-AUC | Accuracy |
|-------|----------|----------|---------|----------|
| GAT | 0.788 | **0.634** | 0.909 | 0.620 |
| **GCN** | **0.797** | 0.578 | **0.915** | **0.633** |

### Key Insights
- **ESM2 dominates handcrafted features:** +16% F1 Micro, +10% ROC-AUC — the protein language model captures far richer biological information than amino acid composition alone.
- **Graph structure helps F1 Macro significantly:** GAT achieves 0.634 F1 Macro vs FFN's 0.581, showing the SSN helps with rarer GO term predictions even though FFN wins on micro-averaged metrics.
- **GCN slightly edges out GAT:** With uniformly weighted MMseqs2 edges, simple neighborhood averaging (GCN) performs slightly better than attention (GAT). GAT would likely benefit from edge-weighted attention using alignment scores.
- **Denser graphs (t=30%) yield best ROC-AUC (0.909):** More edges provide more signal for message passing; sparser graphs (t=70%) boost F1 Macro for minority classes.

---

## Pipeline Overview

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `data/download_swissprot.py` | Stream proteins from UniProt/Swiss-Prot with GO annotations |
| 2 | `similarity/run_mmseqs.py` | All-vs-all pairwise alignment via MMseqs2 (or k-mer Jaccard fallback) |
| 3 | `graph/build_ssn.py` | Build SSN with NetworkX; compute degree, clustering, betweenness, Louvain communities |
| 4a | `features/esm2_embeddings.py` | Generate 320-dim ESM2 embeddings (GPU-accelerated) |
| 4b | `features/handcrafted_features.py` | Compute 26-dim biochemical features (AA composition, MW, pI, GRAVY, etc.) |
| 5 | `training/train.py` | Train GAT/GCN/FFN with PyTorch Geometric; early stopping & checkpointing |

## Ablation Experiments

Run via `--ablation` flag. Four experiments are performed automatically:

1. **GNN vs FFN** — Isolates the contribution of SSN graph structure
2. **ESM2 vs Handcrafted** — Protein language model vs manual biochemical features
3. **Threshold Sweep** (30%, 50%, 70%) — Effect of graph density on performance
4. **GAT vs GCN** — Multi-head attention vs simple neighborhood averaging

## Key Tools

| Tool | Purpose | Version |
|------|---------|---------|
| **MMseqs2** | Ultra-fast protein sequence similarity search | Latest |
| **ESM2** | Meta's protein language model (`esm2_t6_8M_UR50D`, 320-dim) | HuggingFace |
| **NetworkX** | Graph construction, analysis, and community detection | 3.x |
| **PyTorch Geometric** | GNN training framework (GATConv, GCNConv) | 2.x |
| **BioPython** | Biochemical feature extraction | 1.x |

## Project Structure

```
potein/
├── config.py                # All tunable parameters (thresholds, dims, LR, etc.)
├── run_pipeline.py          # End-to-end CLI orchestrator
├── requirements.txt         # Python dependencies
├── data/                    # Stage 1: UniProt data collection
├── similarity/              # Stage 2: MMseqs2 / k-mer similarity
├── graph/                   # Stage 3: SSN construction + graph features
├── features/                # Stage 4: ESM2 + handcrafted feature generation
├── models/                  # GAT, GCN, FFN model architectures
│   ├── gat_model.py         #   Graph Attention Network (multi-head)
│   ├── gcn_model.py         #   Graph Convolutional Network
│   └── baseline_ffn.py      #   Feed-forward baseline (no graph)
├── training/                # Training loop + evaluation metrics
├── experiments/             # Ablation study orchestration
├── results/                 # Output: metrics JSON, training curves, comparison charts
└── utils/                   # Logging, seeding, I/O helpers
```

## Configuration

All parameters are centralized in `config.py`:

| Category | Parameters |
|----------|-----------|
| **Data** | `SUBSET_SIZE`, `TARGET_GO_TERMS`, `MAX_SEQ_LENGTH` |
| **Similarity** | `MMSEQS_BINARY`, `IDENTITY_THRESHOLDS`, `EVALUE_THRESHOLD` |
| **Features** | `ESM2_MODEL_NAME`, `ESM2_BATCH_SIZE`, `DEVICE` |
| **GAT** | `GAT_HIDDEN_DIM=128`, `GAT_NUM_HEADS=4`, `GAT_NUM_LAYERS=2` |
| **GCN** | `GCN_HIDDEN_DIM=128`, `GCN_NUM_LAYERS=2` |
| **Training** | `LEARNING_RATE=1e-3`, `NUM_EPOCHS=200`, `EARLY_STOPPING_PATIENCE=20` |

Set `USE_CUDA=1` environment variable to enable GPU acceleration.
