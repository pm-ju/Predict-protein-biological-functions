"""
Stage 5 — Training Loop
Convert graph data to PyTorch Geometric format, split nodes, and train models.
"""

import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import log, set_seed
from models.gat_model import GATModel
from models.gcn_model import GCNModel
from models.baseline_ffn import FFNModel


def _build_pyg_data(
    embeddings: np.ndarray,
    labels: np.ndarray,
    accessions: list[str],
    edges_csv: Path,
    identity_threshold: float = None,
) -> Data:
    """
    Convert numpy arrays and edge list into a PyTorch Geometric Data object.
    """
    identity_threshold = identity_threshold or config.EDGE_WEIGHT_THRESHOLD

    # Accession → index mapping
    acc_to_idx = {acc: i for i, acc in enumerate(accessions)}

    # Build edge index
    src_list, dst_list, weights = [], [], []
    with open(edges_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s, t = row["source"], row["target"]
            if s in acc_to_idx and t in acc_to_idx:
                si, ti = acc_to_idx[s], acc_to_idx[t]
                identity = float(row["identity"])
                if identity >= identity_threshold:
                    # Undirected: add both directions
                    src_list.extend([si, ti])
                    dst_list.extend([ti, si])
                    weights.extend([identity, identity])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

    x = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.num_nodes = len(accessions)

    log.info(f"PyG Data: {data.num_nodes} nodes, {edge_index.shape[1]} edges, "
             f"features={x.shape[1]}, classes={y.shape[1]}")
    return data


def _create_masks(num_nodes: int, seed: int = None) -> tuple[torch.Tensor, ...]:
    """Create train/val/test boolean masks for node-level splitting."""
    seed = seed or config.RANDOM_SEED
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_nodes)

    train_end = int(config.TRAIN_RATIO * num_nodes)
    val_end = train_end + int(config.VAL_RATIO * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    return train_mask, val_mask, test_mask


def get_model(model_name: str, in_channels: int) -> torch.nn.Module:
    """Factory function to create a model by name."""
    if model_name == "gat":
        return GATModel(in_channels=in_channels)
    elif model_name == "gcn":
        return GCNModel(in_channels=in_channels)
    elif model_name == "ffn":
        return FFNModel(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(
    embeddings: np.ndarray,
    labels: np.ndarray,
    accessions: list[str],
    edges_csv: Path,
    model_name: str = "gat",
    identity_threshold: float = None,
    num_epochs: int = None,
    lr: float = None,
    save_name: str = None,
) -> dict:
    """
    Full training pipeline for a specified model.
    
    Returns a dict with training history and final metrics.
    """
    set_seed(config.RANDOM_SEED)

    num_epochs = num_epochs or config.NUM_EPOCHS
    lr = lr or config.LEARNING_RATE
    save_name = save_name or model_name

    device = torch.device(config.DEVICE)

    # Build PyG data
    data = _build_pyg_data(embeddings, labels, accessions, edges_csv, identity_threshold)
    train_mask, val_mask, test_mask = _create_masks(data.num_nodes)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data = data.to(device)

    # Create model
    model = get_model(model_name, in_channels=data.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    log.info(f"Training {model_name.upper()} model ({sum(p.numel() for p in model.parameters())} params) "
             f"for {num_epochs} epochs…")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.binary_cross_entropy_with_logits(
                val_out[data.val_mask], data.y[data.val_mask]
            ).item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:3d}/{num_epochs}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            ckpt_path = config.MODELS_DIR / f"{save_name}_best.pt"
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                log.info(f"  Early stopping at epoch {epoch} (patience={config.EARLY_STOPPING_PATIENCE})")
                break

    # ── Load best and evaluate on test set ────────────────────────────────
    ckpt_path = config.MODELS_DIR / f"{save_name}_best.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    model.eval()
    with torch.no_grad():
        preds, probs = model.predict(data.x, data.edge_index)

    results = {
        "model": model_name,
        "history": history,
        "data": data,
        "model_obj": model,
        "preds": preds.cpu().numpy(),
        "probs": probs.cpu().numpy(),
        "train_mask": train_mask.numpy(),
        "val_mask": val_mask.numpy(),
        "test_mask": test_mask.numpy(),
        "labels": labels,
    }

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 5: Train a GNN model")
    parser.add_argument("--model", choices=["gat", "gcn", "ffn"], default="gat")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    args = parser.parse_args()

    # Load pre-computed data
    embeddings = np.load(config.FEATURES_DIR / "esm2_embeddings.npy")
    labels = np.load(config.DATA_DIR / "labels.npy")

    order_file = config.FEATURES_DIR / "protein_order.csv"
    accessions = []
    with open(order_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            accessions.append(row[0])

    edges_csv = config.SIMILARITY_DIR / f"edges_t{int(config.DEFAULT_IDENTITY_THRESHOLD*100)}.csv"
    results = train_model(embeddings, labels, accessions, edges_csv, model_name=args.model, num_epochs=args.epochs)
