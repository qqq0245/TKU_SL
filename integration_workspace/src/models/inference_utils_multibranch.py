from __future__ import annotations

from pathlib import Path

import torch

from config import CONFIG
from src.landmarks.feature_builder import resolve_feature_spec
from src.models.inference_utils import apply_confidence_threshold
from src.models.feature_slices import require_multibranch_mode, split_feature_tensor
from src.models.multibranch_model import MultiBranchSignModel


def load_multibranch_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    require_multibranch_mode(checkpoint["feature_mode"])
    checkpoint["feature_spec"] = resolve_feature_spec(
        feature_mode=checkpoint["feature_mode"],
        feature_spec=checkpoint.get("feature_spec"),
        total_dim=checkpoint.get("input_dim"),
    )
    skeleton_branch_type = checkpoint.get("skeleton_branch_type")
    if skeleton_branch_type is None:
        skeleton_branch_type = "gcn" if checkpoint.get("use_gcn_skeleton", False) else "lstm"
    gcn_hidden_dims = tuple(checkpoint.get("gcn_hidden_dims", tuple(CONFIG.gcn_hidden_dims)))
    model = MultiBranchSignModel(
        skeleton_dim=checkpoint["skeleton_dim"],
        location_dim=checkpoint["location_dim"],
        motion_dim=checkpoint["motion_dim"],
        num_classes=checkpoint["num_classes"],
        skeleton_hidden_dim=checkpoint.get("skeleton_branch_hidden_dim", CONFIG.skeleton_branch_hidden_dim),
        location_hidden_dim=checkpoint.get("location_branch_hidden_dim", CONFIG.location_branch_hidden_dim),
        motion_hidden_dim=checkpoint.get("motion_branch_hidden_dim", CONFIG.motion_branch_hidden_dim),
        fusion_hidden_dim=checkpoint.get("fusion_hidden_dim", CONFIG.fusion_hidden_dim),
        dropout=checkpoint.get("dropout", CONFIG.multibranch_dropout),
        bidirectional=checkpoint.get("bidirectional", CONFIG.multibranch_bidirectional),
        skeleton_branch_type=skeleton_branch_type,
        gcn_hidden_dims=gcn_hidden_dims,
        stgcn_hidden_dim=checkpoint.get("stgcn_hidden_dim", CONFIG.stgcn_hidden_dim),
        stgcn_block_channels=tuple(checkpoint.get("stgcn_block_channels", tuple(CONFIG.stgcn_block_channels))),
        stgcn_dropout=checkpoint.get("stgcn_dropout", CONFIG.stgcn_dropout),
        stgcn_use_residual=checkpoint.get("stgcn_use_residual", CONFIG.stgcn_use_residual),
        stgcn_temporal_kernel_size=checkpoint.get(
            "stgcn_temporal_kernel_size",
            CONFIG.stgcn_temporal_kernel_size,
        ),
        location_dropout_prob=checkpoint.get("location_dropout_prob", CONFIG.location_dropout_prob),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_multibranch_sequence(
    model,
    sequence_tensor: torch.Tensor,
    index_to_label: dict[int, str],
    feature_mode: str,
    feature_spec: dict | None = None,
):
    feature_dim = int(sequence_tensor.shape[-1])
    expected_dim = getattr(model, "expected_feature_dim", None)
    if expected_dim is not None and feature_dim != expected_dim:
        raise RuntimeError(
            f"Sequence feature_dim={feature_dim} does not match model expected_feature_dim={expected_dim}. "
            "Please use a checkpoint and dataset captured with the same feature_mode."
        )
    parts = split_feature_tensor(sequence_tensor, feature_mode, feature_spec=feature_spec)
    logits = model(
        parts["skeleton_stream"],
        parts["location_stream"],
        parts["motion_stream"],
    )
    probs = torch.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probs, dim=-1)
    return index_to_label[int(prediction.item())], float(confidence.item()), probs.squeeze(0).cpu().numpy()


__all__ = [
    "load_multibranch_checkpoint",
    "predict_multibranch_sequence",
    "apply_confidence_threshold",
]
