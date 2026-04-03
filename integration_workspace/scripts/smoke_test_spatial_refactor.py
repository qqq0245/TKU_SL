from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.dataset_reader import _apply_spatial_translation_to_sequence
from src.landmarks.feature_builder import build_frame_feature, get_feature_spec
from src.models.feature_slices import get_feature_slices, split_feature_tensor
from src.models.multibranch_model import MultiBranchSignModel


def _build_dummy_frame(frame_index: int) -> dict[str, np.ndarray]:
    left_hand = np.zeros((CONFIG.left_hand_nodes, CONFIG.channels), dtype=np.float32)
    right_hand = np.zeros((CONFIG.right_hand_nodes, CONFIG.channels), dtype=np.float32)
    pose = np.zeros((len(CONFIG.pose_indices), CONFIG.channels), dtype=np.float32)
    left_mask = np.ones((CONFIG.left_hand_nodes,), dtype=np.float32)
    right_mask = np.ones((CONFIG.right_hand_nodes,), dtype=np.float32)
    pose_mask = np.ones((len(CONFIG.pose_indices),), dtype=np.float32)
    mouth_mask = np.ones((1,), dtype=np.float32)
    chin_mask = np.ones((1,), dtype=np.float32)

    for idx in range(CONFIG.left_hand_nodes):
        left_hand[idx] = np.array([0.40 + idx * 0.002, 0.45 + idx * 0.001 + frame_index * 0.002, -0.01 * idx], dtype=np.float32)
        right_hand[idx] = np.array([0.62 + idx * 0.002, 0.44 + idx * 0.0015 + frame_index * 0.002, -0.01 * idx], dtype=np.float32)

    pose[0] = np.array([0.50, 0.28, 0.0], dtype=np.float32)  # nose
    pose[1] = np.array([0.42, 0.38, 0.0], dtype=np.float32)  # left shoulder
    pose[2] = np.array([0.58, 0.38, 0.0], dtype=np.float32)  # right shoulder
    pose[3] = np.array([0.39, 0.48, 0.0], dtype=np.float32)
    pose[4] = np.array([0.61, 0.48, 0.0], dtype=np.float32)
    pose[5] = np.array([0.37, 0.60, 0.0], dtype=np.float32)
    pose[6] = np.array([0.63, 0.60, 0.0], dtype=np.float32)
    pose[7] = np.array([0.44, 0.72, 0.0], dtype=np.float32)
    pose[8] = np.array([0.56, 0.72, 0.0], dtype=np.float32)

    mouth_center = np.array([0.50, 0.33, 0.0], dtype=np.float32)
    chin = np.array([0.50, 0.39, 0.0], dtype=np.float32)
    return build_frame_feature(
        left_hand=left_hand,
        right_hand=right_hand,
        pose=pose,
        mouth_center=mouth_center,
        chin=chin,
        left_hand_mask=left_mask,
        right_hand_mask=right_mask,
        pose_mask=pose_mask,
        mouth_mask=mouth_mask,
        chin_mask=chin_mask,
        feature_mode=CONFIG.feature_mode,
    )


def main() -> None:
    frames = [_build_dummy_frame(frame_index) for frame_index in range(4)]
    sequence = np.stack([frame["feature_vector"] for frame in frames], axis=0).astype(np.float32)
    spec = get_feature_spec(CONFIG.feature_mode)
    assert sequence.shape == (4, spec["total_dim"]), f"Unexpected feature shape: {sequence.shape}"

    augmented = _apply_spatial_translation_to_sequence(
        sequence,
        feature_mode=CONFIG.feature_mode,
        offset_x=0.10,
        offset_y=-0.08,
    )
    slices = get_feature_slices(CONFIG.feature_mode, total_dim=int(sequence.shape[-1]))
    skeleton_before = sequence[:, slices.skeleton_stream]
    skeleton_after = augmented[:, slices.skeleton_stream]
    location_before = sequence[:, slices.location_stream]
    location_after = augmented[:, slices.location_stream]

    assert np.allclose(skeleton_before, skeleton_after), "Skeleton stream should remain translation invariant."
    assert not np.allclose(location_before, location_after), "Location stream should change under spatial augmentation."

    model = MultiBranchSignModel(
        skeleton_dim=slices.skeleton_dim,
        location_dim=slices.location_dim,
        motion_dim=slices.motion_dim,
        num_classes=3,
        skeleton_hidden_dim=CONFIG.skeleton_branch_hidden_dim,
        location_hidden_dim=CONFIG.location_branch_hidden_dim,
        motion_hidden_dim=CONFIG.motion_branch_hidden_dim,
        fusion_hidden_dim=CONFIG.fusion_hidden_dim,
        dropout=CONFIG.multibranch_dropout,
        bidirectional=CONFIG.multibranch_bidirectional,
        skeleton_branch_type=CONFIG.skeleton_branch_type,
        gcn_hidden_dims=CONFIG.gcn_hidden_dims,
        stgcn_hidden_dim=CONFIG.stgcn_hidden_dim,
        stgcn_block_channels=CONFIG.stgcn_block_channels,
        stgcn_dropout=CONFIG.stgcn_dropout,
        stgcn_use_residual=CONFIG.stgcn_use_residual,
        stgcn_temporal_kernel_size=CONFIG.stgcn_temporal_kernel_size,
        location_dropout_prob=1.0,
    )
    model.train()

    batch = torch.from_numpy(np.stack([sequence, augmented], axis=0))
    streams = split_feature_tensor(batch, CONFIG.feature_mode)
    logits = model(
        streams["skeleton_stream"],
        streams["location_stream"],
        streams["motion_stream"],
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert logits.shape == (2, 3), f"Unexpected logits shape: {tuple(logits.shape)}"
    assert model.classifier[-1].weight.grad is not None, "Backward pass did not produce classifier gradients."

    print("Smoke test passed")
    print(f"feature_dim={sequence.shape[-1]} skeleton_dim={slices.skeleton_dim} location_dim={slices.location_dim} motion_dim={slices.motion_dim}")
    print(f"logits_shape={tuple(logits.shape)} loss={float(loss.item()):.6f}")


if __name__ == "__main__":
    main()
