from __future__ import annotations

import numpy as np


POSE_NAME_TO_INDEX = {
    "nose": 0,
    "left_shoulder": 1,
    "right_shoulder": 2,
    "left_elbow": 3,
    "right_elbow": 4,
    "left_wrist": 5,
    "right_wrist": 6,
    "left_hip": 7,
    "right_hip": 8,
}


def safe_mean(points: list[np.ndarray]) -> np.ndarray:
    if not points:
        return np.zeros((3,), dtype=np.float32)
    return np.mean(np.stack(points, axis=0), axis=0).astype(np.float32)


def compute_torso_center(pose: np.ndarray, pose_mask: np.ndarray) -> np.ndarray:
    candidates = []
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        idx = POSE_NAME_TO_INDEX[name]
        if idx < len(pose_mask) and pose_mask[idx] > 0:
            candidates.append(pose[idx])
    if candidates:
        return np.mean(np.stack(candidates, axis=0), axis=0)
    return np.zeros((3,), dtype=np.float32)


def compute_torso_scale(pose: np.ndarray, pose_mask: np.ndarray) -> float:
    left_idx = POSE_NAME_TO_INDEX["left_shoulder"]
    right_idx = POSE_NAME_TO_INDEX["right_shoulder"]
    if pose_mask[left_idx] > 0 and pose_mask[right_idx] > 0:
        width = np.linalg.norm(pose[left_idx] - pose[right_idx])
        if width > 1e-6:
            return float(width)
    return 1.0


def normalize_points(points: np.ndarray, mask: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.zeros_like(points, dtype=np.float32)
    valid = mask > 0
    normalized[valid] = (points[valid] - origin) / scale
    return normalized


def normalize_frame_landmarks(
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
) -> dict[str, np.ndarray]:
    origin = compute_torso_center(pose, pose_mask)
    scale = compute_torso_scale(pose, pose_mask)
    left_hand_center = safe_mean([point for point, valid in zip(left_hand, left_hand_mask) if valid > 0])
    right_hand_center = safe_mean([point for point, valid in zip(right_hand, right_hand_mask) if valid > 0])

    nose = pose[POSE_NAME_TO_INDEX["nose"]] if pose_mask[POSE_NAME_TO_INDEX["nose"]] > 0 else np.zeros((3,), dtype=np.float32)
    left_shoulder = pose[POSE_NAME_TO_INDEX["left_shoulder"]] if pose_mask[POSE_NAME_TO_INDEX["left_shoulder"]] > 0 else np.zeros((3,), dtype=np.float32)
    right_shoulder = pose[POSE_NAME_TO_INDEX["right_shoulder"]] if pose_mask[POSE_NAME_TO_INDEX["right_shoulder"]] > 0 else np.zeros((3,), dtype=np.float32)
    left_hip = pose[POSE_NAME_TO_INDEX["left_hip"]] if pose_mask[POSE_NAME_TO_INDEX["left_hip"]] > 0 else np.zeros((3,), dtype=np.float32)
    right_hip = pose[POSE_NAME_TO_INDEX["right_hip"]] if pose_mask[POSE_NAME_TO_INDEX["right_hip"]] > 0 else np.zeros((3,), dtype=np.float32)
    shoulder_center = safe_mean([point for point in (left_shoulder, right_shoulder) if np.linalg.norm(point) > 0])
    hip_center = safe_mean([point for point in (left_hip, right_hip) if np.linalg.norm(point) > 0])
    chest_center = safe_mean([point for point in (shoulder_center, hip_center) if np.linalg.norm(point) > 0])
    torso_center = origin.astype(np.float32)

    normalized_refs = {
        "nose": ((nose - origin) / scale).astype(np.float32),
        "left_shoulder": ((left_shoulder - origin) / scale).astype(np.float32),
        "right_shoulder": ((right_shoulder - origin) / scale).astype(np.float32),
        "shoulder_center": ((shoulder_center - origin) / scale).astype(np.float32),
        "chest_center": ((chest_center - origin) / scale).astype(np.float32),
        "torso_center": ((torso_center - origin) / scale).astype(np.float32),
        "mouth_center": ((mouth_center - origin) / scale).astype(np.float32) if mouth_mask[0] > 0 else np.zeros((3,), dtype=np.float32),
        "chin": ((chin - origin) / scale).astype(np.float32) if chin_mask[0] > 0 else np.zeros((3,), dtype=np.float32),
    }

    return {
        "origin": origin.astype(np.float32),
        "scale": np.array([scale], dtype=np.float32),
        "left_hand": normalize_points(left_hand, left_hand_mask, origin, scale),
        "right_hand": normalize_points(right_hand, right_hand_mask, origin, scale),
        "pose": normalize_points(pose, pose_mask, origin, scale),
        "left_hand_center": ((left_hand_center - origin) / scale).astype(np.float32) if np.any(left_hand_mask > 0) else np.zeros((3,), dtype=np.float32),
        "right_hand_center": ((right_hand_center - origin) / scale).astype(np.float32) if np.any(right_hand_mask > 0) else np.zeros((3,), dtype=np.float32),
        "mouth_center": normalized_refs["mouth_center"],
        "chin": normalized_refs["chin"],
        "left_hand_mask": left_hand_mask.astype(np.float32),
        "right_hand_mask": right_hand_mask.astype(np.float32),
        "pose_mask": pose_mask.astype(np.float32),
        "mouth_mask": mouth_mask.astype(np.float32),
        "chin_mask": chin_mask.astype(np.float32),
        "left_hand_valid": np.array([float(np.any(left_hand_mask > 0))], dtype=np.float32),
        "right_hand_valid": np.array([float(np.any(right_hand_mask > 0))], dtype=np.float32),
        "reference_points": normalized_refs,
    }
