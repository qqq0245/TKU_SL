from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.landmarks.feature_builder import resolve_feature_spec


@dataclass(frozen=True)
class StreamSlices:
    skeleton_stream: slice
    location_stream: slice | None
    motion_stream: slice | None
    skeleton_dim: int
    location_dim: int
    motion_dim: int


def get_feature_slices(
    feature_mode: str,
    feature_spec: dict | None = None,
    total_dim: int | None = None,
) -> StreamSlices:
    spec = resolve_feature_spec(
        feature_mode=feature_mode,
        feature_spec=feature_spec,
        total_dim=total_dim,
    )
    components = spec["components"]
    skeleton_range = components["landmarks"]
    location_range = components.get("location")
    motion_range = components.get("motion")

    return StreamSlices(
        skeleton_stream=slice(skeleton_range["start"], skeleton_range["end"]),
        location_stream=slice(location_range["start"], location_range["end"]) if location_range else None,
        motion_stream=slice(motion_range["start"], motion_range["end"]) if motion_range else None,
        skeleton_dim=skeleton_range["end"] - skeleton_range["start"],
        location_dim=(location_range["end"] - location_range["start"]) if location_range else 0,
        motion_dim=(motion_range["end"] - motion_range["start"]) if motion_range else 0,
    )


def _slice_like(x: Any, part_slice: slice | None):
    if part_slice is None:
        return None
    if isinstance(x, torch.Tensor):
        return x[..., part_slice]
    if isinstance(x, np.ndarray):
        return x[..., part_slice]
    raise TypeError(f"Unsupported type for split_feature_tensor: {type(x)!r}")


def split_feature_tensor(
    x: torch.Tensor | np.ndarray,
    feature_mode: str,
    feature_spec: dict | None = None,
):
    total_dim = int(x.shape[-1])
    slices = get_feature_slices(feature_mode, feature_spec=feature_spec, total_dim=total_dim)
    return {
        "skeleton_stream": _slice_like(x, slices.skeleton_stream),
        "location_stream": _slice_like(x, slices.location_stream),
        "motion_stream": _slice_like(x, slices.motion_stream),
        "slices": slices,
    }


def require_multibranch_mode(feature_mode: str) -> None:
    supported = {"landmarks_plus_location_motion"}
    if feature_mode not in supported:
        raise RuntimeError(
            f"feature_mode={feature_mode} is not supported by multibranch model. "
            f"Supported modes: {sorted(supported)}"
        )
