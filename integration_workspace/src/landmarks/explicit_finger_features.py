from __future__ import annotations

import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _safe_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def compute_hand_finger_states(hand: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
    states = np.zeros((len(FINGER_NAMES),), dtype=np.float32)
    if hand.shape[0] < 21 or hand_mask.shape[0] < 21 or hand_mask[0] <= 0:
        return states

    wrist = hand[0]
    for finger_index, finger_name in enumerate(FINGER_NAMES):
        chain = FINGER_CHAINS[finger_name]
        if any(hand_mask[joint_idx] <= 0 for joint_idx in chain):
            continue
        tip_idx = chain[-1]
        joint_indices = (0, *chain)
        total_length = 0.0
        for start_idx, end_idx in zip(joint_indices[:-1], joint_indices[1:]):
            total_length += _safe_norm(hand[end_idx] - hand[start_idx])
        if total_length <= 1e-6:
            continue
        tip_distance = _safe_norm(hand[tip_idx] - wrist)
        states[finger_index] = np.clip(tip_distance / total_length, 0.0, 1.0)
    return states


def build_explicit_finger_state_features(normalized: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    left_states = compute_hand_finger_states(normalized["left_hand"], normalized["left_hand_mask"])
    right_states = compute_hand_finger_states(normalized["right_hand"], normalized["right_hand_mask"])
    feature_vector = np.concatenate([left_states, right_states], axis=0).astype(np.float32)
    return {
        "left_hand_finger_states": left_states,
        "right_hand_finger_states": right_states,
        "feature_vector": feature_vector,
    }
