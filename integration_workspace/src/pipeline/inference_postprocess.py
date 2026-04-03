from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from config import CONFIG


@dataclass
class InferenceDecision:
    final_label: str
    final_confidence: float
    raw_label: str
    raw_confidence: float
    status: str
    frame_valid: bool
    sequence_valid: bool
    valid_ratio: float
    smoothed_label: str


class InferencePostprocessor:
    def __init__(self) -> None:
        self.history: deque[tuple[str, float]] = deque(maxlen=CONFIG.smoothing_window)

    def frame_is_valid(self, frame_valid_mask: np.ndarray | None) -> bool:
        if frame_valid_mask is None or len(frame_valid_mask) < 3:
            return False
        left_valid = bool(frame_valid_mask[0] > 0)
        right_valid = bool(frame_valid_mask[1] > 0)
        pose_valid = bool(frame_valid_mask[2] > 0)
        if CONFIG.enable_valid_hand_gate:
            return (left_valid or right_valid) and pose_valid
        return True

    def sequence_validity(self, frame_valid_sequence: np.ndarray | None) -> tuple[bool, float]:
        if frame_valid_sequence is None or len(frame_valid_sequence) == 0:
            return False, 0.0
        valid_flags = [self.frame_is_valid(frame_mask) for frame_mask in frame_valid_sequence]
        ratio = float(np.mean(valid_flags)) if valid_flags else 0.0
        if CONFIG.enable_valid_hand_gate and ratio < CONFIG.min_valid_frame_ratio:
            return False, ratio
        return True, ratio

    def _smooth(self, label: str, confidence: float) -> tuple[str, float, str]:
        if not CONFIG.enable_temporal_smoothing:
            return label, confidence, "raw"

        self.history.append((label, confidence))
        counts = Counter(item[0] for item in self.history)
        majority_label, majority_count = counts.most_common(1)[0]
        majority_confidences = [conf for pred, conf in self.history if pred == majority_label]
        smoothed_confidence = float(sum(majority_confidences) / max(len(majority_confidences), 1))

        if majority_count < CONFIG.stable_min_count and majority_label not in {CONFIG.no_sign_label, CONFIG.unknown_label}:
            return majority_label, smoothed_confidence, "warming_up"
        return majority_label, smoothed_confidence, "smoothed"

    def decide(
        self,
        raw_label: str,
        raw_confidence: float,
        frame_valid_mask: np.ndarray | None,
        frame_valid_sequence: np.ndarray | None,
        thresholded_label: str,
        threshold_status: str,
    ) -> InferenceDecision:
        frame_valid = self.frame_is_valid(frame_valid_mask)
        sequence_valid, valid_ratio = self.sequence_validity(frame_valid_sequence)

        if not frame_valid:
            self.history.append((CONFIG.insufficient_signal_label, 0.0))
            return InferenceDecision(
                final_label=CONFIG.insufficient_signal_label,
                final_confidence=0.0,
                raw_label=raw_label,
                raw_confidence=raw_confidence,
                status="invalid_frame",
                frame_valid=False,
                sequence_valid=sequence_valid,
                valid_ratio=valid_ratio,
                smoothed_label=CONFIG.insufficient_signal_label,
            )

        if not sequence_valid:
            self.history.append((CONFIG.insufficient_signal_label, 0.0))
            return InferenceDecision(
                final_label=CONFIG.insufficient_signal_label,
                final_confidence=0.0,
                raw_label=raw_label,
                raw_confidence=raw_confidence,
                status="insufficient_signal",
                frame_valid=True,
                sequence_valid=False,
                valid_ratio=valid_ratio,
                smoothed_label=CONFIG.insufficient_signal_label,
            )

        smoothed_label, smoothed_confidence, smoothing_status = self._smooth(thresholded_label, raw_confidence)
        final_status = threshold_status if threshold_status != "accepted" else smoothing_status
        return InferenceDecision(
            final_label=smoothed_label,
            final_confidence=smoothed_confidence,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            status=final_status,
            frame_valid=True,
            sequence_valid=True,
            valid_ratio=valid_ratio,
            smoothed_label=smoothed_label,
        )
