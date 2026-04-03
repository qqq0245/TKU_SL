from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.landmarks.feature_builder import resolve_feature_spec
from src.models.gcn_skeleton_branch import reshape_skeleton_stream_to_graph
from src.models.skeleton_graph import NODE_FEATURE_DIM, get_skeleton_graph_definition
from src.models.stgcn_layers import STGCNBlock


class STGCNSkeletonBranch(nn.Module):
    def __init__(
        self,
        skeleton_dim: int,
        hidden_dim: int,
        dropout: float,
        bidirectional: bool,
        block_channels: Sequence[int] = (64, 128, 128),
        temporal_kernel_size: int = 9,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        spec = resolve_feature_spec(feature_mode="landmarks_only", total_dim=skeleton_dim)
        graph = get_skeleton_graph_definition()
        adjacency = graph.normalized_adjacency(include_self=True)
        self.input_dim = skeleton_dim
        self.landmark_graph_dim = spec["landmark_graph_dim"]
        self.explicit_finger_state_dim = spec["explicit_finger_state_dim"]
        self.graph_node_count = graph.node_count
        self.node_feature_dim = NODE_FEATURE_DIM
        self.bidirectional = bidirectional
        self.embedding_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_dim = self.embedding_dim

        channels = [NODE_FEATURE_DIM, *block_channels]
        self.blocks = nn.ModuleList(
            STGCNBlock(
                in_channels=channels[index],
                out_channels=channels[index + 1],
                adjacency=adjacency,
                temporal_kernel_size=temporal_kernel_size,
                dropout=dropout,
                use_residual=use_residual,
            )
            for index in range(len(channels) - 1)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if self.explicit_finger_state_dim > 0:
            self.finger_state_head = nn.Sequential(
                nn.Linear(self.explicit_finger_state_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.finger_state_head = None

    def forward(self, skeleton_stream: torch.Tensor) -> torch.Tensor:
        graph_tensor = reshape_skeleton_stream_to_graph(skeleton_stream)
        x = graph_tensor.permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            x = block(x)
        representation = self.head(x)
        if self.finger_state_head is not None:
            finger_states = skeleton_stream[:, :, self.landmark_graph_dim :]
            finger_summary = finger_states.mean(dim=1)
            representation = representation + self.finger_state_head(finger_summary)
        return representation


__all__ = ["STGCNSkeletonBranch"]
