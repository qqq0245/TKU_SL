from __future__ import annotations

import numpy as np

from config import CONFIG
from src.landmarks.feature_builder import resolve_feature_spec
from src.models.feature_slices import get_feature_slices
from src.models.skeleton_graph import get_skeleton_graph_definition


def infer_node_count_from_feature_dim(feature_dim: int) -> dict[str, int]:
    spec = resolve_feature_spec(feature_mode=CONFIG.feature_mode)
    left = CONFIG.left_hand_nodes
    right = CONFIG.right_hand_nodes
    pose = len(CONFIG.pose_indices)
    expected = spec["landmark_graph_dim"]
    if feature_dim < expected:
        raise ValueError(f"Unexpected feature_dim={feature_dim}, expected at least landmarks_dim={expected}")
    return {"left_hand": left, "right_hand": right, "pose": pose}


def sequence_to_nodes(sequence: np.ndarray) -> dict[str, np.ndarray]:
    counts = infer_node_count_from_feature_dim(int(sequence.shape[-1]))
    pose_count = counts["pose"]
    left_count = counts["left_hand"]
    right_count = counts["right_hand"]

    cursor = 0
    left = sequence[:, cursor : cursor + left_count * 3].reshape(sequence.shape[0], left_count, 3)
    cursor += left_count * 3
    right = sequence[:, cursor : cursor + right_count * 3].reshape(sequence.shape[0], right_count, 3)
    cursor += right_count * 3
    pose = sequence[:, cursor : cursor + pose_count * 3].reshape(sequence.shape[0], pose_count, 3)

    return {"pose": pose.astype(np.float32), "left_hand": left.astype(np.float32), "right_hand": right.astype(np.float32)}


def to_gcn_tensor(sequence: np.ndarray) -> np.ndarray:
    nodes = sequence_to_nodes(sequence)
    combined = np.concatenate([nodes["pose"], nodes["left_hand"], nodes["right_hand"]], axis=1)
    return np.transpose(combined, (2, 0, 1))[:, :, :, np.newaxis].astype(np.float32)


def export_sample_to_gcn_dict(sequence: np.ndarray, metadata: dict) -> dict:
    feature_slices = get_feature_slices(metadata.get("feature_mode", CONFIG.feature_mode))
    gcn_tensor = to_gcn_tensor(sequence)
    graph = get_skeleton_graph_definition()
    return {
        "sample_id": metadata["sample_id"],
        "label": metadata["label"],
        "baseline_sequence_shape": list(sequence.shape),
        "gcn_tensor_shape": list(gcn_tensor.shape),
        "feature_mode": metadata.get("feature_mode"),
        "streams": {
            "skeleton_stream": {
                "source": "normalized landmark section only",
                "tensor_shape": list(gcn_tensor.shape),
                "graph_ready_for": ["gcn", "stgcn"],
            },
            "location_stream": {
                "source": "location feature section",
                "index_range": (
                    None
                    if feature_slices.location_stream is None
                    else {"start": feature_slices.location_stream.start, "end": feature_slices.location_stream.stop}
                ),
                "included_in_gcn": False,
            },
            "motion_stream": {
                "source": "motion feature section",
                "index_range": (
                    None
                    if feature_slices.motion_stream is None
                    else {"start": feature_slices.motion_stream.start, "end": feature_slices.motion_stream.stop}
                ),
                "included_in_gcn": False,
            },
        },
        "mapping": {
            "pose_indices": list(CONFIG.pose_indices),
            "pose_node_count": len(CONFIG.pose_indices),
            "left_hand_node_count": CONFIG.left_hand_nodes,
            "right_hand_node_count": CONFIG.right_hand_nodes,
            "combined_order": ["pose", "left_hand", "right_hand"],
            "graph_node_count": graph.node_count,
            "graph_edge_count": len(graph.edges),
            "target_format": "C,T,V,M",
            "stgcn_internal_format": "B,C,T,V",
            "landmarks_component_range": {
                "start": feature_slices.skeleton_stream.start,
                "end": feature_slices.skeleton_stream.stop,
            },
        },
    }
