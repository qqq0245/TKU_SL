from __future__ import annotations

import torch
from torch import nn


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.projection = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape (B, T, V, C), received {tuple(x.shape)}")
        if adjacency.dim() != 2:
            raise ValueError(f"Expected adjacency with shape (V, V), received {tuple(adjacency.shape)}")

        aggregated = torch.einsum("vw,btwc->btvc", adjacency, x)
        projected = self.projection(aggregated)
        normalized = self.norm(projected)
        return self.dropout(self.activation(normalized))


__all__ = ["GraphConv"]
