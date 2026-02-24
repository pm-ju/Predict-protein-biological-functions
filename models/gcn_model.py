"""
Graph Convolutional Network (GCN) — baseline GNN without attention.
Uses simple neighbourhood averaging instead of learned attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GCNModel(nn.Module):
    """
    Multi-layer Graph Convolutional Network for multi-label node classification.
    
    Architecture:
      Input → [GCNConv + BN + ReLU + Dropout] × (num_layers-1) → GCNConv
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        out_channels: int = None,
        num_layers: int = None,
        dropout: float = None,
    ):
        super().__init__()

        hidden_channels = hidden_channels or config.GCN_HIDDEN_DIM
        out_channels = out_channels or config.NUM_GO_CLASSES
        num_layers = num_layers or config.GCN_NUM_LAYERS
        dropout = dropout or config.GCN_DROPOUT

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        # Final layer
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x

    def predict(self, x, edge_index, threshold: float = 0.5):
        logits = self.forward(x, edge_index)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float(), probs
