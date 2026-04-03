from __future__ import annotations

import torch
from torch import nn


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape (B, C, T, V), received {tuple(x.shape)}")
        if adjacency.dim() != 2:
            raise ValueError(f"Expected adjacency with shape (V, V), received {tuple(adjacency.shape)}")
        projected = self.projection(x)
        return torch.einsum("vw,bctw->bctv", adjacency, projected)


class TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9, dropout: float = 0.0) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2, 0)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal(x)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        temporal_kernel_size: int = 9,
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.spatial = SpatialGraphConv(in_channels, out_channels)
        self.spatial_norm = nn.BatchNorm2d(out_channels)
        self.temporal = TemporalConv(out_channels, kernel_size=temporal_kernel_size, dropout=dropout)
        self.activation = nn.ReLU(inplace=True)
        self.register_buffer("adjacency", adjacency)

        if not use_residual:
            self.residual = None
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = 0
        if self.use_residual:
            residual = self.residual(x) if self.residual is not None else 0
        x = self.spatial(x, self.adjacency)
        x = self.spatial_norm(x)
        x = self.temporal(x)
        x = x + residual
        return self.activation(x)


__all__ = ["SpatialGraphConv", "TemporalConv", "STGCNBlock"]
