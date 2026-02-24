"""
Global configuration for the GNN Protein Function Prediction pipeline.
All tunable parameters are centralised here.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "output"
SIMILARITY_DIR = PROJECT_ROOT / "similarity" / "output"
GRAPH_DIR = PROJECT_ROOT / "graph" / "output"
FEATURES_DIR = PROJECT_ROOT / "features" / "output"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create output directories
for d in [DATA_DIR, SIMILARITY_DIR, GRAPH_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Stage 1: Data Collection ─────────────────────────────────────────────────
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/stream"
SUBSET_SIZE = 1000                      # Number of proteins to download
TARGET_GO_TERMS = [                     # GO Molecular Function terms to predict
    "GO:0003723",   # RNA binding
    "GO:0005524",   # ATP binding
    "GO:0003677",   # DNA binding
    "GO:0004674",   # protein serine/threonine kinase activity
    "GO:0008270",   # zinc ion binding
    "GO:0003824",   # catalytic activity
    "GO:0016491",   # oxidoreductase activity
    "GO:0004672",   # protein kinase activity
]
NUM_GO_CLASSES = len(TARGET_GO_TERMS)
MAX_SEQ_LENGTH = 1000                   # Truncate sequences longer than this

# ── Stage 2: Similarity Computation ──────────────────────────────────────────
MMSEQS_BINARY = "mmseqs"                # Path to mmseqs2 binary (or just "mmseqs" if in PATH)
IDENTITY_THRESHOLDS = [0.3, 0.5, 0.7]  # For ablation experiments
DEFAULT_IDENTITY_THRESHOLD = 0.3       # Default threshold for main pipeline
EVALUE_THRESHOLD = 1e-5                 # E-value cutoff
USE_MMSEQS_FALLBACK = False            # Set True to force Python fallback even when mmseqs2 is available

# ── Stage 3: Graph Construction ──────────────────────────────────────────────
EDGE_WEIGHT_THRESHOLD = 0.3            # Minimum identity to keep an edge
COMPUTE_CENTRALITY = True              # Betweenness centrality (slow for large graphs)
RUN_COMMUNITY_DETECTION = True         # Louvain community detection

# ── Stage 4: Feature Generation ──────────────────────────────────────────────
ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # Smallest ESM2 (8M params, 320-dim)
ESM2_EMBEDDING_DIM = 320
ESM2_BATCH_SIZE = 16                   # Sequences per batch
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

# Handcrafted features
HANDCRAFTED_DIM = 26                   # 20 AA composition + mol weight + pI + length + arom + instab + gravy

# ── Stage 5: Model & Training ────────────────────────────────────────────────
# GAT configuration
GAT_HIDDEN_DIM = 128
GAT_NUM_HEADS = 4
GAT_NUM_LAYERS = 2
GAT_DROPOUT = 0.3

# GCN configuration
GCN_HIDDEN_DIM = 128
GCN_NUM_LAYERS = 2
GCN_DROPOUT = 0.3

# FFN baseline configuration
FFN_HIDDEN_DIMS = [256, 128]
FFN_DROPOUT = 0.3

# Training
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42
