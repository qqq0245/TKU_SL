from __future__ import annotations

import numpy as np


def _speed(vector: np.ndarray, valid: bool) -> np.ndarray:
    if not valid:
        return np.zeros((1,), dtype=np.float32)
    return np.array([float(np.linalg.norm(vector))], dtype=np.float32)


def _acceleration(curr_velocity: np.ndarray, prev_velocity: np.ndarray | None, valid: bool) -> np.ndarray:
    if not valid or prev_velocity is None:
        return np.zeros((3,), dtype=np.float32)
    return (curr_velocity - prev_velocity).astype(np.float32)


def _unit_direction(vector: np.ndarray, valid: bool) -> np.ndarray:
    if not valid:
        return np.zeros((3,), dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return np.zeros((3,), dtype=np.float32)
    return (vector / norm).astype(np.float32)


def build_motion_features(
    frame_feature: dict[str, np.ndarray],
    prev_feature: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    left_valid = bool(frame_feature["left_hand_valid"][0] > 0)
    right_valid = bool(frame_feature["right_hand_valid"][0] > 0)
    prev_left_velocity = prev_feature["motion_components"]["left_velocity"] if prev_feature and "motion_components" in prev_feature else None
    prev_right_velocity = prev_feature["motion_components"]["right_velocity"] if prev_feature and "motion_components" in prev_feature else None

    left_velocity = frame_feature["left_hand_center"] - prev_feature["left_hand_center"] if prev_feature is not None and left_valid else np.zeros((3,), dtype=np.float32)
    right_velocity = frame_feature["right_hand_center"] - prev_feature["right_hand_center"] if prev_feature is not None and right_valid else np.zeros((3,), dtype=np.float32)
    left_speed = _speed(left_velocity, left_valid and prev_feature is not None)
    right_speed = _speed(right_velocity, right_valid and prev_feature is not None)

    prev_distance = np.linalg.norm(prev_feature["left_hand_center"] - prev_feature["right_hand_center"]) if prev_feature is not None else 0.0
    curr_distance = np.linalg.norm(frame_feature["left_hand_center"] - frame_feature["right_hand_center"]) if left_valid and right_valid else 0.0
    distance_delta = np.array([curr_distance - prev_distance if prev_feature is not None and left_valid and right_valid else 0.0], dtype=np.float32)

    left_acc = _acceleration(left_velocity, prev_left_velocity, left_valid and prev_feature is not None)
    right_acc = _acceleration(right_velocity, prev_right_velocity, right_valid and prev_feature is not None)
    left_torso_motion = left_velocity.astype(np.float32)
    right_torso_motion = right_velocity.astype(np.float32)
    left_direction = _unit_direction(left_velocity, left_valid and prev_feature is not None)
    right_direction = _unit_direction(right_velocity, right_valid and prev_feature is not None)

    sync = np.array(
        [
            float(np.dot(left_direction, right_direction))
            if left_valid and right_valid and prev_feature is not None
            else 0.0
        ],
        dtype=np.float32,
    )
    validity = np.array([float(left_valid), float(right_valid)], dtype=np.float32)

    feature_vector = np.concatenate(
        [
            left_velocity,
            right_velocity,
            left_speed,
            right_speed,
            distance_delta,
            left_acc,
            right_acc,
            left_torso_motion,
            right_torso_motion,
            left_direction,
            right_direction,
            sync,
            validity,
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "feature_vector": feature_vector,
        "left_velocity": left_velocity.astype(np.float32),
        "right_velocity": right_velocity.astype(np.float32),
        "left_acceleration": left_acc.astype(np.float32),
        "right_acceleration": right_acc.astype(np.float32),
        "distance_delta": distance_delta.astype(np.float32),
        "sync": sync.astype(np.float32),
    }
