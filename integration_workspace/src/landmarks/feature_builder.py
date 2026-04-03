from __future__ import annotations

import numpy as np

from config import CONFIG
from .explicit_finger_features import FINGER_NAMES, build_explicit_finger_state_features
from .location_features import build_location_features
from .motion_features import build_motion_features
from .normalization import normalize_frame_landmarks


FEATURE_MODES = {
    "landmarks_only",
    "landmarks_plus_location",
    "landmarks_plus_location_motion",
}


def _localize_hand_to_wrist(hand: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
    localized = np.zeros_like(hand, dtype=np.float32)
    if hand.shape[0] == 0 or hand_mask.shape[0] == 0 or hand_mask[0] <= 0:
        return localized
    wrist = hand[0].astype(np.float32)
    valid = hand_mask > 0
    localized[valid] = hand[valid] - wrist
    return localized


def _mid_shoulder_anchor(pose: np.ndarray, pose_mask: np.ndarray) -> np.ndarray:
    left_idx = 1
    right_idx = 2
    if pose_mask[left_idx] > 0 and pose_mask[right_idx] > 0:
        return ((pose[left_idx] + pose[right_idx]) / 2.0).astype(np.float32)
    if pose_mask[left_idx] > 0:
        return pose[left_idx].astype(np.float32)
    if pose_mask[right_idx] > 0:
        return pose[right_idx].astype(np.float32)
    return np.zeros((3,), dtype=np.float32)


def _localize_pose_to_mid_shoulder(pose: np.ndarray, pose_mask: np.ndarray) -> np.ndarray:
    localized = np.zeros_like(pose, dtype=np.float32)
    anchor = _mid_shoulder_anchor(pose, pose_mask)
    valid = pose_mask > 0
    localized[valid] = pose[valid] - anchor
    return localized


def _localize_pose(pose: np.ndarray, pose_mask: np.ndarray, anchor_mode: str) -> np.ndarray:
    mode = str(anchor_mode).strip().lower()
    if mode == "mid_shoulder":
        return _localize_pose_to_mid_shoulder(pose, pose_mask)
    if mode == "torso_center":
        return pose.astype(np.float32, copy=True)
    raise ValueError(f"Unsupported pose_local_anchor={anchor_mode}")


def _landmark_graph_dim() -> int:
    return (CONFIG.left_hand_nodes + CONFIG.right_hand_nodes + len(CONFIG.pose_indices)) * CONFIG.channels + (
        CONFIG.left_hand_nodes + CONFIG.right_hand_nodes + len(CONFIG.pose_indices)
    )


def _explicit_finger_state_dim(include_explicit_finger_states: bool) -> int:
    if not include_explicit_finger_states:
        return 0
    return CONFIG.explicit_finger_state_dim_per_hand * 2


def _location_total(include_zone_encoding: bool) -> int:
    location_vector_dim = 13 * 3
    zone_dim = len(CONFIG.location_zone_names) * 2 if include_zone_encoding else 0
    location_validity_dim = 4
    return location_vector_dim + zone_dim + location_validity_dim


def _motion_total() -> int:
    return 3 + 3 + 1 + 1 + 1 + 3 + 3 + 3 + 3 + 3 + 3 + 1 + 2


def _build_landmark_feature_vector(normalized: dict[str, np.ndarray], spec: dict) -> np.ndarray:
    feature_parts = [
        normalized["left_hand_local"].reshape(-1),
        normalized["right_hand_local"].reshape(-1),
        normalized["pose_local"].reshape(-1),
        normalized["left_hand_mask"],
        normalized["right_hand_mask"],
        normalized["pose_mask"],
    ]
    if spec["explicit_finger_state_dim"] > 0:
        feature_parts.append(normalized["explicit_finger_state_features"])
    return np.concatenate(feature_parts, axis=0).astype(np.float32)


def _range(start: int, length: int) -> dict[str, int]:
    return {"start": start, "end": start + length}


def get_feature_spec(
    feature_mode: str | None = None,
    include_zone_encoding: bool | None = None,
    include_explicit_finger_states: bool | None = None,
) -> dict:
    feature_mode = feature_mode or CONFIG.feature_mode
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode={feature_mode}")

    include_zone_encoding = CONFIG.use_zone_encoding if include_zone_encoding is None else include_zone_encoding
    include_explicit_finger_states = (
        CONFIG.enable_explicit_finger_states
        if include_explicit_finger_states is None
        else include_explicit_finger_states
    )

    graph_dim = _landmark_graph_dim()
    explicit_finger_state_dim = _explicit_finger_state_dim(include_explicit_finger_states)
    landmarks_total = graph_dim + explicit_finger_state_dim
    location_total = _location_total(include_zone_encoding)
    motion_total = _motion_total()

    components = {}
    cursor = 0
    components["landmarks"] = _range(cursor, landmarks_total)
    cursor += landmarks_total
    if feature_mode in {"landmarks_plus_location", "landmarks_plus_location_motion"}:
        components["location"] = _range(cursor, location_total)
        cursor += location_total
    if feature_mode == "landmarks_plus_location_motion":
        components["motion"] = _range(cursor, motion_total)
        cursor += motion_total

    landmark_sections = {
        "graph": _range(0, graph_dim),
        "explicit_finger_states": _range(graph_dim, explicit_finger_state_dim) if explicit_finger_state_dim > 0 else None,
    }
    finger_state_names = [
        f"{hand}_{finger_name}"
        for hand in ("left", "right")
        for finger_name in FINGER_NAMES
    ]

    return {
        "feature_mode": feature_mode,
        "feature_version": "explicit_finger_states_v1" if include_explicit_finger_states else "legacy_v1",
        "include_explicit_finger_states": include_explicit_finger_states,
        "landmark_graph_dim": graph_dim,
        "explicit_finger_state_dim": explicit_finger_state_dim,
        "landmarks_dim": landmarks_total,
        "location_dim": location_total,
        "motion_dim": motion_total,
        "total_dim": cursor,
        "components": components,
        "landmark_sections": landmark_sections,
        "finger_state_names": finger_state_names[:explicit_finger_state_dim],
        "zone_names": list(CONFIG.location_zone_names),
    }


def infer_feature_spec(
    *,
    feature_mode: str,
    total_dim: int,
    include_zone_encoding: bool | None = None,
) -> dict:
    include_zone_encoding = CONFIG.use_zone_encoding if include_zone_encoding is None else include_zone_encoding
    location_total = _location_total(include_zone_encoding)
    motion_total = _motion_total()
    graph_dim = _landmark_graph_dim()

    landmark_total = total_dim
    if feature_mode in {"landmarks_plus_location", "landmarks_plus_location_motion"}:
        landmark_total -= location_total
    if feature_mode == "landmarks_plus_location_motion":
        landmark_total -= motion_total

    explicit_finger_state_dim = landmark_total - graph_dim
    if explicit_finger_state_dim not in {0, CONFIG.explicit_finger_state_dim_per_hand * 2}:
        raise ValueError(
            f"Unable to infer explicit finger state layout from feature_mode={feature_mode} total_dim={total_dim}"
        )
    return get_feature_spec(
        feature_mode=feature_mode,
        include_zone_encoding=include_zone_encoding,
        include_explicit_finger_states=explicit_finger_state_dim > 0,
    )


def resolve_feature_spec(
    *,
    feature_mode: str,
    feature_spec: dict | None = None,
    total_dim: int | None = None,
    include_zone_encoding: bool | None = None,
) -> dict:
    if feature_spec is not None:
        spec = dict(feature_spec)
        if "explicit_finger_state_dim" not in spec:
            if total_dim is None:
                total_dim = int(spec.get("total_dim", 0))
            spec = infer_feature_spec(
                feature_mode=feature_mode,
                total_dim=total_dim,
                include_zone_encoding=include_zone_encoding,
            )
        return spec
    if total_dim is not None:
        return infer_feature_spec(
            feature_mode=feature_mode,
            total_dim=total_dim,
            include_zone_encoding=include_zone_encoding,
        )
    return get_feature_spec(
        feature_mode=feature_mode,
        include_zone_encoding=include_zone_encoding,
    )


def build_frame_feature(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    pose: np.ndarray,
    mouth_center: np.ndarray,
    chin: np.ndarray,
    left_hand_mask: np.ndarray,
    right_hand_mask: np.ndarray,
    pose_mask: np.ndarray,
    mouth_mask: np.ndarray,
    chin_mask: np.ndarray,
    prev_feature: dict[str, np.ndarray] | None = None,
    feature_mode: str | None = None,
    feature_spec: dict | None = None,
    pose_local_anchor: str = "mid_shoulder",
) -> dict[str, np.ndarray]:
    feature_mode = feature_mode or CONFIG.feature_mode
    spec = resolve_feature_spec(feature_mode=feature_mode, feature_spec=feature_spec)

    normalized = normalize_frame_landmarks(
        left_hand=left_hand,
        right_hand=right_hand,
        pose=pose,
        mouth_center=mouth_center,
        chin=chin,
        left_hand_mask=left_hand_mask,
        right_hand_mask=right_hand_mask,
        pose_mask=pose_mask,
        mouth_mask=mouth_mask,
        chin_mask=chin_mask,
    )

    finger_states = build_explicit_finger_state_features(normalized)
    normalized.update(finger_states)
    normalized["explicit_finger_state_features"] = finger_states["feature_vector"]
    normalized["left_hand_local"] = _localize_hand_to_wrist(normalized["left_hand"], normalized["left_hand_mask"])
    normalized["right_hand_local"] = _localize_hand_to_wrist(normalized["right_hand"], normalized["right_hand_mask"])
    normalized["pose_local"] = _localize_pose(
        normalized["pose"],
        normalized["pose_mask"],
        anchor_mode=pose_local_anchor,
    )
    landmarks_vector = _build_landmark_feature_vector(normalized, spec)
    location = build_location_features(normalized)
    motion = build_motion_features(normalized, prev_feature=prev_feature)

    feature_parts = [landmarks_vector]
    if feature_mode in {"landmarks_plus_location", "landmarks_plus_location_motion"}:
        feature_parts.append(location["feature_vector"])
    if feature_mode == "landmarks_plus_location_motion":
        feature_parts.append(motion["feature_vector"])

    frame_valid_mask = np.array(
        [
            float(np.any(left_hand_mask > 0)),
            float(np.any(right_hand_mask > 0)),
            float(np.any(pose_mask > 0)),
            float(mouth_mask[0] > 0),
            float(chin_mask[0] > 0),
        ],
        dtype=np.float32,
    )

    normalized["mouth_center_raw"] = mouth_center.astype(np.float32)
    normalized["chin_raw"] = chin.astype(np.float32)
    normalized["landmark_features"] = landmarks_vector
    normalized["location_features"] = location["feature_vector"]
    normalized["motion_features"] = motion["feature_vector"]
    normalized["location_components"] = location
    normalized["motion_components"] = motion
    normalized["feature_vector"] = np.concatenate(feature_parts, axis=0).astype(np.float32)
    normalized["feature_mode"] = feature_mode
    normalized["feature_spec"] = spec
    normalized["pose_local_anchor"] = str(pose_local_anchor).strip().lower()
    normalized["frame_valid_mask"] = frame_valid_mask
    return normalized
