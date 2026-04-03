from __future__ import annotations

import torch
from torch import nn

from config import CONFIG
from src.models.gcn_skeleton_branch import GCNSkeletonBranch
from src.models.stgcn_skeleton_branch import STGCNSkeletonBranch


class BranchEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.temporal_hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.frame_encoder(x)
        sequence_output, _ = self.temporal_encoder(encoded)
        return sequence_output[:, -1, :]


class MultiBranchSignModel(nn.Module):
    def __init__(
        self,
        skeleton_dim: int,
        location_dim: int,
        motion_dim: int,
        num_classes: int,
        skeleton_hidden_dim: int,
        location_hidden_dim: int,
        motion_hidden_dim: int,
        fusion_hidden_dim: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
        skeleton_branch_type: str = "lstm",
        gcn_hidden_dims: tuple[int, ...] = (64, 128),
        stgcn_hidden_dim: int | None = None,
        stgcn_block_channels: tuple[int, ...] = (64, 128, 128),
        stgcn_dropout: float = 0.3,
        stgcn_use_residual: bool = True,
        stgcn_temporal_kernel_size: int = 9,
        use_gcn_skeleton: bool | None = None,
        location_dropout_prob: float = CONFIG.location_dropout_prob,
    ) -> None:
        super().__init__()
        self.skeleton_dim = skeleton_dim
        self.location_dim = location_dim
        self.motion_dim = motion_dim
        self.expected_feature_dim = skeleton_dim + location_dim + motion_dim
        if use_gcn_skeleton is not None:
            skeleton_branch_type = "gcn" if use_gcn_skeleton else "lstm"
        if skeleton_branch_type not in {"lstm", "gcn", "stgcn"}:
            raise ValueError(f"Unsupported skeleton_branch_type={skeleton_branch_type}")
        self.skeleton_branch_type = skeleton_branch_type
        self.use_gcn_skeleton = skeleton_branch_type in {"gcn", "stgcn"}
        self.location_dropout_prob = float(location_dropout_prob)
        if skeleton_branch_type == "gcn":
            self.skeleton_branch = GCNSkeletonBranch(
                skeleton_dim=skeleton_dim,
                hidden_dim=skeleton_hidden_dim,
                dropout=dropout,
                bidirectional=bidirectional,
                gcn_hidden_dims=gcn_hidden_dims,
            )
        elif skeleton_branch_type == "stgcn":
            self.skeleton_branch = STGCNSkeletonBranch(
                skeleton_dim=skeleton_dim,
                hidden_dim=stgcn_hidden_dim or skeleton_hidden_dim,
                dropout=stgcn_dropout,
                bidirectional=bidirectional,
                block_channels=stgcn_block_channels,
                temporal_kernel_size=stgcn_temporal_kernel_size,
                use_residual=stgcn_use_residual,
            )
        else:
            self.skeleton_branch = BranchEncoder(
                input_dim=skeleton_dim,
                projection_dim=skeleton_hidden_dim,
                hidden_dim=skeleton_hidden_dim,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        self.location_branch = BranchEncoder(
            input_dim=location_dim,
            projection_dim=location_hidden_dim,
            hidden_dim=location_hidden_dim,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.motion_branch = BranchEncoder(
            input_dim=motion_dim,
            projection_dim=motion_hidden_dim,
            hidden_dim=motion_hidden_dim,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        fusion_input_dim = (
            self.skeleton_branch.output_dim
            + self.location_branch.output_dim
            + self.motion_branch.output_dim
        )
        self.fusion_input_dim = fusion_input_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )
        self.last_branch_outputs: dict[str, torch.Tensor] = {}

    def forward(
        self,
        skeleton_stream: torch.Tensor,
        location_stream: torch.Tensor,
        motion_stream: torch.Tensor,
    ) -> torch.Tensor:
        skeleton_repr = self.skeleton_branch(skeleton_stream)
        location_repr_pre_dropout = self.location_branch(location_stream)
        location_repr = location_repr_pre_dropout
        if self.training and self.location_dropout_prob > 0.0:
            drop_batch = bool(torch.rand((), device=location_repr.device) < self.location_dropout_prob)
            if drop_batch:
                location_repr = torch.zeros_like(location_repr)
        motion_repr = self.motion_branch(motion_stream)
        self.last_branch_outputs = {
            "skeleton_repr": skeleton_repr,
            "location_repr_pre_dropout": location_repr_pre_dropout,
            "location_repr": location_repr,
            "motion_repr": motion_repr,
        }
        fused = torch.cat([skeleton_repr, location_repr, motion_repr], dim=-1)
        return self.classifier(fused)
