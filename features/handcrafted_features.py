"""
Stage 4b — Handcrafted Biochemical Features
Compute traditional protein properties as an alternative to ESM2 embeddings.
"""

import csv
import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import log
from data.download_swissprot import load_sequences


# Amino acid letters (standard 20)
AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


def _compute_aa_composition(sequence: str) -> list[float]:
    """Compute the fraction of each of the 20 standard amino acids."""
    length = max(len(sequence), 1)
    counts = {aa: 0 for aa in AA_LETTERS}
    for aa in sequence.upper():
        if aa in counts:
            counts[aa] += 1
    return [counts[aa] / length for aa in AA_LETTERS]


def _compute_features_biopython(sequence: str) -> dict:
    """Compute biochemical features using BioPython's ProteinAnalysis."""
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        # Filter to standard amino acids only
        clean_seq = "".join(aa for aa in sequence.upper() if aa in AA_LETTERS)
        if len(clean_seq) < 5:
            return None

        analysis = ProteinAnalysis(clean_seq)
        return {
            "molecular_weight": analysis.molecular_weight(),
            "isoelectric_point": analysis.isoelectric_point(),
            "aromaticity": analysis.aromaticity(),
            "instability_index": analysis.instability_index(),
            "gravy": analysis.gravy(),
        }
    except Exception as e:
        log.warning(f"BioPython analysis failed: {e}")
        return None


def generate_handcrafted_features(
    sequences: dict[str, str] = None,
    output_dir: Path = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute biochemical features for all proteins.
    
    Feature vector (26-dim):
      - 20 amino acid composition fractions
      - molecular weight
      - isoelectric point
      - sequence length (log-scaled)
      - aromaticity
      - instability index
      - GRAVY (grand average of hydropathy)
    
    Returns (features_array [N, 26], ordered_accessions).
    """
    output_dir = output_dir or config.FEATURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    features_npy = output_dir / "handcrafted_features.npy"

    if features_npy.exists():
        log.info(f"Handcrafted features already exist. Loading…")
        return np.load(features_npy), None

    if sequences is None:
        sequences = load_sequences()

    accessions = list(sequences.keys())
    feature_list = []

    log.info(f"Computing handcrafted features for {len(accessions)} proteins…")

    for acc in accessions:
        seq = sequences[acc]
        aa_comp = _compute_aa_composition(seq)
        bio_feats = _compute_features_biopython(seq)

        if bio_feats is None:
            # Fallback: dummy values
            bio_feats = {
                "molecular_weight": 0.0,
                "isoelectric_point": 7.0,
                "aromaticity": 0.0,
                "instability_index": 40.0,
                "gravy": 0.0,
            }

        features = aa_comp + [
            bio_feats["molecular_weight"] / 100000.0,  # Normalise
            bio_feats["isoelectric_point"] / 14.0,       # Normalise to [0, 1]
            np.log1p(len(seq)) / 10.0,                   # Log-scaled length
            bio_feats["aromaticity"],
            bio_feats["instability_index"] / 100.0,
            (bio_feats["gravy"] + 5.0) / 10.0,           # Shift and scale
        ]
        feature_list.append(features)

    features_arr = np.array(feature_list, dtype=np.float32)
    np.save(features_npy, features_arr)

    log.info(f"Saved handcrafted features ({features_arr.shape}) to {features_npy}")
    return features_arr, accessions


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_handcrafted_features()
