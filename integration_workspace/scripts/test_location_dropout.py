from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.landmarks.feature_builder import get_feature_spec
from src.models.feature_slices import get_feature_slices, split_feature_tensor
from src.models.multibranch_model import MultiBranchSignModel


def _dummy_batch(batch_size: int = 2, sequence_length: int = 4) -> torch.Tensor:
    spec = get_feature_spec(CONFIG.feature_mode)
    values = torch.linspace(
        0.05,
        1.25,
        steps=batch_size * sequence_length * spec["total_dim"],
        dtype=torch.float32,
    )
    return values.view(batch_size, sequence_length, spec["total_dim"])


def _build_model(location_dropout_prob: float) -> tuple[MultiBranchSignModel, object]:
    spec = get_feature_spec(CONFIG.feature_mode)
    slices = get_feature_slices(CONFIG.feature_mode, feature_spec=spec, total_dim=spec["total_dim"])
    model = MultiBranchSignModel(
        skeleton_dim=slices.skeleton_dim,
        location_dim=slices.location_dim,
        motion_dim=slices.motion_dim,
        num_classes=5,
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
        location_dropout_prob=location_dropout_prob,
    )
    return model, slices


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().abs().sum().item())
    return total


def test_training_mode() -> None:
    model, _ = _build_model(location_dropout_prob=1.0)
    model.train()
    batch = _dummy_batch()
    streams = split_feature_tensor(batch, CONFIG.feature_mode)
    logits = model(
        streams["skeleton_stream"],
        streams["location_stream"],
        streams["motion_stream"],
    )
    branch = model.last_branch_outputs

    assert torch.count_nonzero(branch["skeleton_repr"]).item() > 0, "skeleton branch output should not be zero"
    assert torch.count_nonzero(branch["motion_repr"]).item() > 0, "motion branch output should not be zero"
    assert torch.count_nonzero(branch["location_repr_pre_dropout"]).item() > 0, "location branch pre-dropout should not be zero"
    assert torch.count_nonzero(branch["location_repr"]).item() == 0, "location branch output should be fully zero in training mode"

    labels = torch.tensor([0, 1], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    location_grad = _grad_norm(model.location_branch.parameters())
    skeleton_grad = _grad_norm(model.skeleton_branch.parameters())
    motion_grad = _grad_norm(model.motion_branch.parameters())
    classifier_grad = _grad_norm(model.classifier.parameters())

    assert classifier_grad > 0.0, "classifier gradients should be non-zero"
    assert skeleton_grad > 0.0, "skeleton branch gradients should be non-zero"
    assert motion_grad > 0.0, "motion branch gradients should be non-zero"
    assert location_grad == 0.0, "location branch gradients should be zero when dropout is forced"


def test_evaluation_mode() -> None:
    model, _ = _build_model(location_dropout_prob=1.0)
    model.eval()
    batch = _dummy_batch()
    with torch.no_grad():
        streams = split_feature_tensor(batch, CONFIG.feature_mode)
        _ = model(
            streams["skeleton_stream"],
            streams["location_stream"],
            streams["motion_stream"],
        )
    branch = model.last_branch_outputs
    assert torch.count_nonzero(branch["location_repr"]).item() > 0, "location branch output should be preserved in eval mode"


def main() -> int:
    try:
        test_training_mode()
        print("[PASS] Training mode forced location dropout behaves as expected.")
        test_evaluation_mode()
        print("[PASS] Evaluation mode preserves location branch output.")
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
