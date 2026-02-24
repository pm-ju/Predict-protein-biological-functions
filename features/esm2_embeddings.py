"""
Stage 4a — ESM2 Embeddings
Generate protein sequence embeddings using Meta's ESM2 protein language model.
"""

import csv
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import log
from data.download_swissprot import load_sequences


def generate_esm2_embeddings(
    sequences: dict[str, str] = None,
    output_dir: Path = None,
    batch_size: int = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate ESM2 embeddings for all protein sequences.
    
    Returns (embeddings_array [N, 320], ordered_accessions).
    """
    from transformers import AutoTokenizer, AutoModel

    output_dir = output_dir or config.FEATURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size or config.ESM2_BATCH_SIZE

    embeddings_npy = output_dir / "esm2_embeddings.npy"
    order_file = output_dir / "protein_order.csv"

    if embeddings_npy.exists() and order_file.exists():
        log.info(f"ESM2 embeddings already exist at {embeddings_npy}. Loading…")
        embeddings = np.load(embeddings_npy)
        accessions = []
        with open(order_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                accessions.append(row[0])
        return embeddings, accessions

    if sequences is None:
        sequences = load_sequences()

    device = torch.device(config.DEVICE)
    log.info(f"Loading ESM2 model ({config.ESM2_MODEL_NAME}) on {device}…")

    tokenizer = AutoTokenizer.from_pretrained(config.ESM2_MODEL_NAME)
    model = AutoModel.from_pretrained(config.ESM2_MODEL_NAME).to(device)
    model.eval()

    accessions = list(sequences.keys())
    seqs = [sequences[acc] for acc in accessions]
    all_embeddings = []

    log.info(f"Generating embeddings for {len(seqs)} proteins (batch_size={batch_size})…")

    for i in tqdm(range(0, len(seqs), batch_size), desc="ESM2 embeddings"):
        batch_seqs = seqs[i:i+batch_size]

        # Truncate sequences for ESM2 (max 1024 tokens for small model)
        batch_seqs = [s[:1022] for s in batch_seqs]

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pool over sequence length (ignoring special tokens)
        # outputs.last_hidden_state shape: [batch, seq_len, hidden_dim]
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [batch, seq_len, 1]
        hidden = outputs.last_hidden_state * attention_mask
        embeddings = hidden.sum(dim=1) / attention_mask.sum(dim=1)  # [batch, hidden_dim]

        all_embeddings.append(embeddings.cpu().numpy())

    embeddings_arr = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # Save
    np.save(embeddings_npy, embeddings_arr)

    with open(order_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accession"])
        for acc in accessions:
            writer.writerow([acc])

    log.info(f"Saved ESM2 embeddings ({embeddings_arr.shape}) to {embeddings_npy}")
    return embeddings_arr, accessions


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_esm2_embeddings()
