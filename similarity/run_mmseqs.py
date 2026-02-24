"""
Stage 2 — Similarity Computation
Run MMseqs2 all-vs-all search or fall back to a simple Python-based
pairwise identity computation for small datasets.
"""

import csv
import subprocess
import shutil
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import log
from data.download_swissprot import load_sequences


# ──────────────────────────────────────────────────────────────────────────────
# MMseqs2 runner
# ──────────────────────────────────────────────────────────────────────────────

def _write_fasta(sequences: dict[str, str], path: Path) -> None:
    """Write sequences dict to a FASTA file."""
    with open(path, "w") as f:
        for acc, seq in sequences.items():
            f.write(f">{acc}\n{seq}\n")


def _mmseqs_available() -> bool:
    """Check if mmseqs2 binary is accessible."""
    return shutil.which(config.MMSEQS_BINARY) is not None


def run_mmseqs(
    sequences: dict[str, str],
    identity_threshold: float = None,
    output_dir: Path = None,
) -> Path:
    """
    Run MMseqs2 all-vs-all easy-search.
    Returns path to edges CSV.
    """
    identity_threshold = identity_threshold or config.DEFAULT_IDENTITY_THRESHOLD
    output_dir = output_dir or config.SIMILARITY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_csv = output_dir / f"edges_t{int(identity_threshold*100)}.csv"

    if edges_csv.exists():
        log.info(f"Edge list already exists at {edges_csv}. Skipping.")
        return edges_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = Path(tmpdir) / "proteins.fasta"
        _write_fasta(sequences, fasta_path)

        result_path = Path(tmpdir) / "result.tsv"
        tmp_mmseqs = Path(tmpdir) / "tmp"
        tmp_mmseqs.mkdir()

        cmd = [
            config.MMSEQS_BINARY, "easy-search",
            str(fasta_path), str(fasta_path), str(result_path), str(tmp_mmseqs),
            "--min-seq-id", str(identity_threshold),
            "-e", str(config.EVALUE_THRESHOLD),
            "--format-output", "query,target,pident,evalue,bits",
            "-s", "7.5",
        ]

        log.info(f"Running MMseqs2: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Parse results
        edges = []
        seen = set()
        with open(result_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                query, target = parts[0], parts[1]
                if query == target:
                    continue
                edge_key = tuple(sorted([query, target]))
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                edges.append({
                    "source": query,
                    "target": target,
                    "identity": float(parts[2]),
                    "evalue": float(parts[3]),
                    "bitscore": float(parts[4]),
                })

    # Save
    with open(edges_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "identity", "evalue", "bitscore"])
        writer.writeheader()
        writer.writerows(edges)

    log.info(f"MMseqs2 found {len(edges)} edges (threshold={identity_threshold}). Saved to {edges_csv}")
    return edges_csv


# ──────────────────────────────────────────────────────────────────────────────
# Python fallback — simple k-mer Jaccard similarity
# ──────────────────────────────────────────────────────────────────────────────

def _kmer_set(sequence: str, k: int = 3) -> set[str]:
    """Extract the set of k-mers from a sequence."""
    return {sequence[i:i+k] for i in range(len(sequence) - k + 1)}


def compute_similarity_fallback(
    sequences: dict[str, str],
    identity_threshold: float = None,
    output_dir: Path = None,
) -> Path:
    """
    Compute pairwise k-mer-based Jaccard similarity as an MMseqs2 fallback.
    Much slower but requires no external tools.
    """
    identity_threshold = identity_threshold or config.DEFAULT_IDENTITY_THRESHOLD
    output_dir = output_dir or config.SIMILARITY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_csv = output_dir / f"edges_t{int(identity_threshold*100)}.csv"

    if edges_csv.exists():
        log.info(f"Edge list already exists at {edges_csv}. Skipping.")
        return edges_csv

    log.info(f"Computing pairwise k-mer Jaccard similarity (fallback mode)…")

    accs = list(sequences.keys())
    n = len(accs)

    # Pre-compute k-mer sets
    kmer_sets = {acc: _kmer_set(sequences[acc]) for acc in accs}

    edges = []
    total_pairs = n * (n - 1) // 2
    with tqdm(total=total_pairs, desc="Pairwise similarity") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                a, b = accs[i], accs[j]
                sa, sb = kmer_sets[a], kmer_sets[b]
                if not sa or not sb:
                    pbar.update(1)
                    continue
                jaccard = len(sa & sb) / len(sa | sb)
                if jaccard >= identity_threshold:
                    edges.append({
                        "source": a,
                        "target": b,
                        "identity": round(jaccard, 4),
                        "evalue": 0.0,
                        "bitscore": 0.0,
                    })
                pbar.update(1)

    with open(edges_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "identity", "evalue", "bitscore"])
        writer.writeheader()
        writer.writerows(edges)

    log.info(f"Fallback found {len(edges)} edges (Jaccard≥{identity_threshold}). Saved to {edges_csv}")
    return edges_csv


# ──────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ──────────────────────────────────────────────────────────────────────────────

def compute_similarity(
    sequences: dict[str, str] = None,
    identity_threshold: float = None,
    output_dir: Path = None,
) -> Path:
    """
    Compute pairwise similarity — MMseqs2 if available, else Python fallback.
    Returns path to edges CSV.
    """
    if sequences is None:
        sequences = load_sequences()

    if _mmseqs_available() and not config.USE_MMSEQS_FALLBACK:
        return run_mmseqs(sequences, identity_threshold, output_dir)
    elif _mmseqs_available():
        log.info("MMseqs2 found but fallback mode forced in config.")
        return compute_similarity_fallback(sequences, identity_threshold, output_dir)
    else:
        log.warning("MMseqs2 not found. Using Python k-mer Jaccard fallback (slower).")
        return compute_similarity_fallback(sequences, identity_threshold, output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: Compute pairwise similarity")
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_IDENTITY_THRESHOLD)
    args = parser.parse_args()
    compute_similarity(identity_threshold=args.threshold)
