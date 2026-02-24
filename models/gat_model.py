"""
Graph Attention Network (GAT) — primary model architecture.
Uses multi-head attention to learn which neighbours are most informative
in the Sequence Similarity Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GATModel(nn.Module):
    """
    Multi-layer Graph Attention Network for multi-label node classification.
    
    Architecture:
      Input → [GATConv + BN + ELU + Dropout] × (num_layers-1) → GATConv → Sigmoid
    
    Early layers use multi-head attention; final layer uses single head.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        out_channels: int = None,
        num_heads: int = None,
        num_layers: int = None,
        dropout: float = None,
    ):
        super().__init__()

        hidden_channels = hidden_channels or config.GAT_HIDDEN_DIM
        out_channels = out_channels or config.NUM_GO_CLASSES
        num_heads = num_heads or config.GAT_NUM_HEADS
        num_layers = num_layers or config.GAT_NUM_LAYERS
        dropout = dropout or config.GAT_DROPOUT

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels * num_heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels * num_heads))

        # Final layer (single head, output classes)
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x  # Raw logits — apply sigmoid in the loss/inference

    def predict(self, x, edge_index, threshold: float = 0.5):
        """Forward pass + sigmoid + thresholding for inference."""
        logits = self.forward(x, edge_index)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float(), probs
