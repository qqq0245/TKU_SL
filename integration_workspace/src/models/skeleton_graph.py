from __future__ import annotations

from dataclasses import dataclass

import torch

from config import CONFIG


POSE_NODE_NAMES = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)

HAND_NODE_COUNT = 21
POSE_NODE_COUNT = len(CONFIG.pose_indices)
TOTAL_NODE_COUNT = POSE_NODE_COUNT + HAND_NODE_COUNT + HAND_NODE_COUNT
NODE_FEATURE_DIM = 4

LEFT_HAND_OFFSET = POSE_NODE_COUNT
RIGHT_HAND_OFFSET = POSE_NODE_COUNT + HAND_NODE_COUNT


@dataclass(frozen=True)
class SkeletonGraphDefinition:
    node_names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]

    @property
    def node_count(self) -> int:
        return len(self.node_names)

    def adjacency(self, include_self: bool = True) -> torch.Tensor:
        adjacency = torch.zeros((self.node_count, self.node_count), dtype=torch.float32)
        for source, target in self.edges:
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
        if include_self:
            adjacency += torch.eye(self.node_count, dtype=torch.float32)
        return adjacency

    def normalized_adjacency(self, include_self: bool = True) -> torch.Tensor:
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        adjacency = self.adjacency(include_self=include_self)
        degree = adjacency.sum(dim=1)
        inv_sqrt_degree = torch.pow(degree.clamp(min=1.0), -0.5)
        d_inv_sqrt = torch.diag(inv_sqrt_degree)
        return d_inv_sqrt @ adjacency @ d_inv_sqrt


def _pose_edges() -> tuple[tuple[int, int], ...]:
    return (
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (1, 7),
        (2, 8),
        (7, 8),
    )


def _hand_edges(offset: int) -> tuple[tuple[int, int], ...]:
    hand_edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),
    )
    return tuple((offset + source, offset + target) for source, target in hand_edges)


def get_skeleton_graph_definition() -> SkeletonGraphDefinition:
    # Node order is fixed for all graph-based branches:
    # 1. pose upper-body nodes
    # 2. left hand nodes
    # 3. right hand nodes
    node_names = (
        *POSE_NODE_NAMES,
        *(f"left_hand_{index}" for index in range(HAND_NODE_COUNT)),
        *(f"right_hand_{index}" for index in range(HAND_NODE_COUNT)),
    )
    edges = (
        *_pose_edges(),
        *_hand_edges(LEFT_HAND_OFFSET),
        *_hand_edges(RIGHT_HAND_OFFSET),
        (5, LEFT_HAND_OFFSET + 0),
        (6, RIGHT_HAND_OFFSET + 0),
    )
    return SkeletonGraphDefinition(node_names=node_names, edges=edges)


__all__ = [
    "HAND_NODE_COUNT",
    "LEFT_HAND_OFFSET",
    "NODE_FEATURE_DIM",
    "POSE_NODE_COUNT",
    "POSE_NODE_NAMES",
    "RIGHT_HAND_OFFSET",
    "SkeletonGraphDefinition",
    "TOTAL_NODE_COUNT",
    "get_skeleton_graph_definition",
]
