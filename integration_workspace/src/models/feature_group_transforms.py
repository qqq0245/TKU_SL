from __future__ import annotations

from typing import Any

import numpy as np
import torch

from config import CONFIG


def hand_mask_validity_indices(feature_spec: dict) -> list[int]:
    channels = int(CONFIG.channels)
    left_xyz = int(CONFIG.left_hand_nodes) * channels
    right_xyz = int(CONFIG.right_hand_nodes) * channels
    pose_nodes = len(CONFIG.pose_indices)
    pose_xyz = pose_nodes * channels
    left_mask_start = left_xyz + right_xyz + pose_xyz
    right_mask_start = left_mask_start + int(CONFIG.left_hand_nodes)
    pose_mask_start = right_mask_start + int(CONFIG.right_hand_nodes)
    location_range = feature_spec["components"]["location"]
    motion_range = feature_spec["components"]["motion"]
    location_end = int(location_range["end"])
    motion_end = int(motion_range["end"])
    location_validity_start = location_end - 4
    motion_validity_start = motion_end - 2
    return (
        list(range(left_mask_start, left_mask_start + int(CONFIG.left_hand_nodes)))
        + list(range(right_mask_start, right_mask_start + int(CONFIG.right_hand_nodes)))
        + list(range(location_validity_start, location_end))
        + list(range(motion_validity_start, motion_end))
    )


def pose_hip_coordinate_indices(feature_spec: dict) -> list[int]:
    channels = int(CONFIG.channels)
    left_xyz = int(CONFIG.left_hand_nodes) * channels
    right_xyz = int(CONFIG.right_hand_nodes) * channels
    pose_coord_start = left_xyz + right_xyz
    hip_pose_indices = [7, 8]
    indices: list[int] = []
    for pose_index in hip_pose_indices:
        base = pose_coord_start + pose_index * channels
        indices.extend(range(base, base + channels))
    return indices


def apply_hand_mask_validity_scale(
    sequence: np.ndarray | torch.Tensor,
    *,
    feature_spec: dict,
    scale: float,
) -> np.ndarray | torch.Tensor:
    if abs(float(scale) - 1.0) <= 1e-8:
        return sequence
    indices = hand_mask_validity_indices(feature_spec)
    if isinstance(sequence, np.ndarray):
        adjusted = sequence.copy()
        adjusted[..., indices] *= float(scale)
        return adjusted
    if isinstance(sequence, torch.Tensor):
        adjusted = sequence.clone()
        adjusted[..., indices] = adjusted[..., indices] * float(scale)
        return adjusted
    raise TypeError(f"Unsupported sequence type: {type(sequence)!r}")


def apply_pose_hip_coordinate_scale(
    sequence: np.ndarray | torch.Tensor,
    *,
    feature_spec: dict,
    scale: float,
) -> np.ndarray | torch.Tensor:
    if abs(float(scale) - 1.0) <= 1e-8:
        return sequence
    indices = pose_hip_coordinate_indices(feature_spec)
    if isinstance(sequence, np.ndarray):
        adjusted = sequence.copy()
        adjusted[..., indices] *= float(scale)
        return adjusted
    if isinstance(sequence, torch.Tensor):
        adjusted = sequence.clone()
        adjusted[..., indices] = adjusted[..., indices] * float(scale)
        return adjusted
    raise TypeError(f"Unsupported sequence type: {type(sequence)!r}")
