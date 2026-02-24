"""
Feed-Forward Network (FFN) — baseline that ignores graph structure.
Takes only node features and predicts labels without any message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FFNModel(nn.Module):
    """
    Multi-layer feed-forward network for multi-label node classification.
    This baseline does NOT use graph structure — it serves as the
    ablation control to isolate the contribution of the SSN.
    
    Architecture:
      Input → [Linear + BN + ReLU + Dropout] × N → Linear
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: list[int] = None,
        out_channels: int = None,
        dropout: float = None,
    ):
        super().__init__()

        hidden_dims = hidden_dims or config.FFN_HIDDEN_DIMS
        out_channels = out_channels or config.NUM_GO_CLASSES
        dropout = dropout or config.FFN_DROPOUT

        layers = []
        prev_dim = in_channels
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, out_channels))
        self.network = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        """edge_index is accepted but ignored — keeps API consistent with GNN models."""
        return self.network(x)

    def predict(self, x, edge_index=None, threshold: float = 0.5):
        logits = self.forward(x, edge_index)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float(), probs
