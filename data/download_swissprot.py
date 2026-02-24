"""
Stage 1 — Data Collection
Download a curated subset of reviewed proteins from UniProt/Swiss-Prot,
extract sequences and GO Molecular Function annotations.
"""

import csv
import json
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import log


def _build_query(go_terms: list[str], limit: int) -> str:
    """Build a UniProt REST API query string for reviewed proteins with given GO terms."""
    # UniProt search uses go:NNNNNNN format (strip 'GO:' prefix)
    go_clause = " OR ".join(f"(go:{term.replace('GO:', '')})" for term in go_terms)
    return f"(reviewed:true) AND ({go_clause})"


def download_proteins(
    subset_size: int = None,
    output_dir: Path = None,
) -> tuple[Path, Path]:
    """
    Download proteins from Swiss-Prot with GO Molecular Function annotations.
    
    Returns paths to (proteins_csv, labels_npy).
    """
    subset_size = subset_size or config.SUBSET_SIZE
    output_dir = output_dir or config.DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    proteins_csv = output_dir / "proteins.csv"
    labels_npy = output_dir / "labels.npy"
    metadata_json = output_dir / "metadata.json"

    if proteins_csv.exists() and labels_npy.exists():
        log.info(f"Data already downloaded at {output_dir}. Skipping.")
        return proteins_csv, labels_npy

    log.info(f"Downloading up to {subset_size} proteins from UniProt…")

    # ── Query UniProt REST API ────────────────────────────────────────────
    query = _build_query(config.TARGET_GO_TERMS, subset_size)
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,sequence,organism_name,go_id,go_f",
    }

    log.info(f"Query: {query}")
    response = requests.get(config.UNIPROT_API_URL, params=params, stream=True, timeout=120)
    response.raise_for_status()

    proteins = []
    label_matrix = []

    # ── Parse and filter ──────────────────────────────────────────────────
    lines_iter = response.iter_lines(decode_unicode=True)
    header = next(lines_iter, "").split("\t")

    with tqdm(total=subset_size, desc="Downloading proteins", unit="prot") as pbar:
        for line in lines_iter:
            if not line:
                continue
            
            row = line.split("\t")
            if len(row) < 4:
                continue
                
            accession = row[0].strip()
            sequence = row[1].strip()
            organism = row[2].strip()
            go_ids_raw = row[3].strip() if len(row) > 3 else ""

            if not sequence or len(sequence) < 10:
                continue

            # Truncate long sequences
            if len(sequence) > config.MAX_SEQ_LENGTH:
                sequence = sequence[:config.MAX_SEQ_LENGTH]

            # Parse GO IDs and create multi-label vector
            go_ids = [g.strip() for g in go_ids_raw.replace(";", " ").split() if g.startswith("GO:")]
            label_vec = [1 if term in go_ids else 0 for term in config.TARGET_GO_TERMS]

            # Keep only proteins that have at least one of our target GO terms
            if sum(label_vec) == 0:
                continue

            proteins.append({
                "accession": accession,
                "sequence": sequence,
                "organism": organism,
                "go_ids": ";".join(go_ids),
                "seq_length": len(sequence),
            })
            label_matrix.append(label_vec)
            pbar.update(1)
            
            if len(proteins) >= subset_size:
                break

    response.close()
    log.info(f"Retained {len(proteins)} proteins with target GO annotations.")

    # ── Save outputs ──────────────────────────────────────────────────────
    with open(proteins_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["accession", "sequence", "organism", "go_ids", "seq_length"])
        writer.writeheader()
        writer.writerows(proteins)

    labels_arr = np.array(label_matrix, dtype=np.float32)
    np.save(labels_npy, labels_arr)

    # Save metadata
    meta = {
        "num_proteins": len(proteins),
        "num_go_terms": config.NUM_GO_CLASSES,
        "go_terms": config.TARGET_GO_TERMS,
        "label_distribution": {
            term: int(labels_arr[:, i].sum())
            for i, term in enumerate(config.TARGET_GO_TERMS)
        },
    }
    with open(metadata_json, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Saved proteins to {proteins_csv}")
    log.info(f"Saved labels ({labels_arr.shape}) to {labels_npy}")
    log.info(f"Label distribution: {meta['label_distribution']}")

    return proteins_csv, labels_npy


def load_sequences(proteins_csv: Path = None) -> dict[str, str]:
    """Load protein sequences from the CSV as {accession: sequence}."""
    proteins_csv = proteins_csv or config.DATA_DIR / "proteins.csv"
    sequences = {}
    with open(proteins_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences[row["accession"]] = row["sequence"]
    return sequences


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Download Swiss-Prot data")
    parser.add_argument("--size", type=int, default=config.SUBSET_SIZE, help="Number of proteins")
    args = parser.parse_args()
    download_proteins(subset_size=args.size)
