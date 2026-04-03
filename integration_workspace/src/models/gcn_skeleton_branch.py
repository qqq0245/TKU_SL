from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from config import CONFIG
from src.landmarks.feature_builder import resolve_feature_spec
from src.models.gcn_layers import GraphConv
from src.models.skeleton_graph import NODE_FEATURE_DIM, TOTAL_NODE_COUNT, get_skeleton_graph_definition


def reshape_skeleton_stream_to_graph(skeleton_stream: torch.Tensor) -> torch.Tensor:
    if skeleton_stream.dim() != 3:
        raise ValueError(
            f"Expected skeleton_stream with shape (B, T, D), received {tuple(skeleton_stream.shape)}"
        )

    batch_size, steps, feature_dim = skeleton_stream.shape
    left_nodes = CONFIG.left_hand_nodes
    right_nodes = CONFIG.right_hand_nodes
    pose_nodes = len(CONFIG.pose_indices)
    spec = resolve_feature_spec(feature_mode="landmarks_only", total_dim=feature_dim)
    graph_dim = spec["landmark_graph_dim"]
    if feature_dim < graph_dim:
        raise ValueError(f"Expected skeleton_dim>={graph_dim}, received {feature_dim}")

    cursor = 0
    graph_stream = skeleton_stream[:, :, :graph_dim]
    left_xyz = graph_stream[:, :, cursor : cursor + left_nodes * 3].reshape(batch_size, steps, left_nodes, 3)
    cursor += left_nodes * 3
    right_xyz = graph_stream[:, :, cursor : cursor + right_nodes * 3].reshape(batch_size, steps, right_nodes, 3)
    cursor += right_nodes * 3
    pose_xyz = graph_stream[:, :, cursor : cursor + pose_nodes * 3].reshape(batch_size, steps, pose_nodes, 3)
    cursor += pose_nodes * 3
    left_mask = graph_stream[:, :, cursor : cursor + left_nodes].reshape(batch_size, steps, left_nodes, 1)
    cursor += left_nodes
    right_mask = graph_stream[:, :, cursor : cursor + right_nodes].reshape(batch_size, steps, right_nodes, 1)
    cursor += right_nodes
    pose_mask = graph_stream[:, :, cursor : cursor + pose_nodes].reshape(batch_size, steps, pose_nodes, 1)

    pose_nodes_tensor = torch.cat([pose_xyz, pose_mask], dim=-1)
    left_nodes_tensor = torch.cat([left_xyz, left_mask], dim=-1)
    right_nodes_tensor = torch.cat([right_xyz, right_mask], dim=-1)
    graph_tensor = torch.cat([pose_nodes_tensor, left_nodes_tensor, right_nodes_tensor], dim=2)

    if graph_tensor.shape[2] != TOTAL_NODE_COUNT or graph_tensor.shape[3] != NODE_FEATURE_DIM:
        raise RuntimeError(
            "Unexpected graph tensor shape after reshape: "
            f"{tuple(graph_tensor.shape)} expected V={TOTAL_NODE_COUNT}, C={NODE_FEATURE_DIM}"
        )
    return graph_tensor


class GCNSkeletonBranch(nn.Module):
    def __init__(
        self,
        skeleton_dim: int,
        hidden_dim: int,
        dropout: float,
        bidirectional: bool,
        gcn_hidden_dims: Sequence[int] = (64, 128),
    ) -> None:
        super().__init__()
        spec = resolve_feature_spec(feature_mode="landmarks_only", total_dim=skeleton_dim)

        self.input_dim = skeleton_dim
        self.landmark_graph_dim = spec["landmark_graph_dim"]
        self.explicit_finger_state_dim = spec["explicit_finger_state_dim"]
        graph = get_skeleton_graph_definition()
        self.graph_node_count = graph.node_count
        self.node_feature_dim = NODE_FEATURE_DIM
        self.bidirectional = bidirectional
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

        self.register_buffer("adjacency", graph.adjacency(include_self=True))
        self.register_buffer("normalized_adjacency", graph.normalized_adjacency(include_self=True))

        channels = [self.node_feature_dim, *gcn_hidden_dims]
        self.gcn_layers = nn.ModuleList(
            GraphConv(channels[index], channels[index + 1], dropout=dropout)
            for index in range(len(channels) - 1)
        )
        temporal_input_dim = channels[-1]
        self.temporal_encoder = nn.LSTM(
            input_size=temporal_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        if self.explicit_finger_state_dim > 0:
            self.finger_state_head = nn.Sequential(
                nn.Linear(self.explicit_finger_state_dim, self.output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.finger_state_head = None

    def forward(self, skeleton_stream: torch.Tensor) -> torch.Tensor:
        graph_tensor = reshape_skeleton_stream_to_graph(skeleton_stream)
        x = graph_tensor
        for layer in self.gcn_layers:
            x = layer(x, self.normalized_adjacency)
        pooled = self.dropout(x.mean(dim=2))
        sequence_output, _ = self.temporal_encoder(pooled)
        representation = sequence_output[:, -1, :]
        if self.finger_state_head is not None:
            finger_states = skeleton_stream[:, :, self.landmark_graph_dim :]
            finger_summary = finger_states.mean(dim=1)
            representation = representation + self.finger_state_head(finger_summary)
        return representation


__all__ = ["GCNSkeletonBranch", "reshape_skeleton_stream_to_graph"]
