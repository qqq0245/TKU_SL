from __future__ import annotations

import csv
from dataclasses import dataclass, field
import copy
import importlib.util
import json
import os
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CONFIG
from src.dataset.sequence_builder import SequenceBuilder
from src.landmarks.feature_builder import build_frame_feature, resolve_feature_spec
from src.landmarks.holistic_extractor import FrameLandmarks, HolisticExtractor
from src.models.feature_slices import split_feature_tensor
from src.models.feature_group_transforms import apply_hand_mask_validity_scale, apply_pose_hip_coordinate_scale
from src.models.inference_utils import apply_confidence_threshold
from src.models.inference_utils_multibranch import load_multibranch_checkpoint
from src.pipeline.class_specific_disambiguation import ClassSpecificDisambiguator
from src.pipeline.inference_postprocess import InferenceDecision, InferencePostprocessor


IGNORED_LABELS = {
    CONFIG.no_sign_label,
    CONFIG.unknown_label,
    CONFIG.insufficient_signal_label,
    "collecting",
}
ASSIST_TARGET_LABELS = {"you", "i", "like", "mother"}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
USER_TUNING_SETTINGS_PATH = PROJECT_ROOT / "realtime_tuning_settings.py"
DEFAULT_GESTURE_ASSIST_SETTINGS = {
    "assist_shape_weight": 0.85,
    "assist_position_weight": 0.15,
    "profile_shape_weight": 0.85,
    "profile_position_weight": 0.15,
    "assist_multiplier_base": 0.92,
    "assist_multiplier_scale": 0.18,
    "assist_multiplier_min": 0.92,
    "assist_multiplier_max": 1.10,
}
LEFT_FACE_VECTOR_DIM = 9
LEFT_FACE_NOSE_Z_INDEX = 2
MOTHER_YOU_PROTOTYPE_GROUPS = 6
LEFT_BODY_VECTOR_DIM = 9
LEFT_BODY_CHEST_Y_INDEX = 4
LEFT_BODY_TORSO_Y_INDEX = 7
YOU_I_LEFT_BODY_PROTOTYPE_GROUPS = 6
DEFAULT_YOU_LIKE_FRONTLOAD_POWER = 1.64
DEFAULT_YOU_LIKE_PAIRWISE_DELTA = 0.98
LIKE_I_LOCATION_PROTOTYPE_GROUPS = 8
DEFAULT_LIKE_I_PAIRWISE_DELTA = 2.80
DEFAULT_WANT_ANCHOR_FALLBACK_MAX_NOSIGN_CONFIDENCE = 0.95
DEFAULT_FATHER_TRIGGER_RESCUE_PAIRWISE_DELTA = 6.50
DEFAULT_FATHER_TRIGGER_RESCUE_MIN_RATIO = 0.75
DEFAULT_FATHER_TRIGGER_RESCUE_MAX_NOSIGN_CONFIDENCE = 0.995


def _coerce_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def load_runtime_inference_flags() -> dict[str, object]:
    flags = {
        "enable_rule_based_disambiguation": True,
        "enable_dynamic_nosign_suppression": False,
        "nosign_penalty_factor": 1.0,
        "trigger_patience": 2,
        "idle_patience": 5,
        "min_action_frames": 10,
        "min_motion_energy": 0.020,
        "emit_confidence_threshold": 0.40,
        "min_top_margin": 0.40,
        "pre_context_frames": 5,
        "max_buffer_frames": 60,
        "hand_mask_validity_scale": 1.0,
        "pose_hip_coordinate_scale": 1.0,
        "pose_local_anchor": "mid_shoulder",
        "enable_mother_nose_z_calibration": False,
        "mother_nose_z_offset": -1.5,
        "enable_you_i_leftbody_calibration": False,
        "you_i_leftbody_y_offset": -0.035,
        "enable_you_like_pairwise_calibration": False,
        "you_like_pairwise_delta": DEFAULT_YOU_LIKE_PAIRWISE_DELTA,
        "enable_like_i_pairwise_calibration": False,
        "like_i_pairwise_delta": DEFAULT_LIKE_I_PAIRWISE_DELTA,
        "enable_you_like_frontloaded_rescore": False,
        "you_like_frontload_power": DEFAULT_YOU_LIKE_FRONTLOAD_POWER,
        "enable_want_anchor_fallback": False,
        "want_anchor_fallback_max_nosign_confidence": DEFAULT_WANT_ANCHOR_FALLBACK_MAX_NOSIGN_CONFIDENCE,
        "enable_father_trigger_rescue": False,
        "father_trigger_rescue_pairwise_delta": DEFAULT_FATHER_TRIGGER_RESCUE_PAIRWISE_DELTA,
        "father_trigger_rescue_min_ratio": DEFAULT_FATHER_TRIGGER_RESCUE_MIN_RATIO,
        "father_trigger_rescue_max_nosign_confidence": DEFAULT_FATHER_TRIGGER_RESCUE_MAX_NOSIGN_CONFIDENCE,
    }
    if not USER_TUNING_SETTINGS_PATH.exists():
        return flags
    try:
        spec = importlib.util.spec_from_file_location("user_realtime_tuning_settings", USER_TUNING_SETTINGS_PATH)
        if spec is None or spec.loader is None:
            return flags
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "ENABLE_RULE_BASED_DISAMBIGUATION"):
            flags["enable_rule_based_disambiguation"] = bool(
                getattr(module, "ENABLE_RULE_BASED_DISAMBIGUATION")
            )
        if hasattr(module, "ENABLE_DYNAMIC_NOSIGN_SUPPRESSION"):
            flags["enable_dynamic_nosign_suppression"] = bool(
                getattr(module, "ENABLE_DYNAMIC_NOSIGN_SUPPRESSION")
            )
        if hasattr(module, "NOSIGN_PENALTY_FACTOR"):
            flags["nosign_penalty_factor"] = float(getattr(module, "NOSIGN_PENALTY_FACTOR"))
        if hasattr(module, "TRIGGER_PATIENCE"):
            flags["trigger_patience"] = int(getattr(module, "TRIGGER_PATIENCE"))
        if hasattr(module, "IDLE_PATIENCE"):
            flags["idle_patience"] = int(getattr(module, "IDLE_PATIENCE"))
        if hasattr(module, "MIN_ACTION_FRAMES"):
            flags["min_action_frames"] = int(getattr(module, "MIN_ACTION_FRAMES"))
        if hasattr(module, "MIN_MOTION_ENERGY"):
            flags["min_motion_energy"] = float(getattr(module, "MIN_MOTION_ENERGY"))
        if hasattr(module, "EMIT_CONFIDENCE_THRESHOLD"):
            flags["emit_confidence_threshold"] = float(getattr(module, "EMIT_CONFIDENCE_THRESHOLD"))
        if hasattr(module, "MIN_TOP_MARGIN"):
            flags["min_top_margin"] = float(getattr(module, "MIN_TOP_MARGIN"))
        if hasattr(module, "PRE_CONTEXT_FRAMES"):
            flags["pre_context_frames"] = int(getattr(module, "PRE_CONTEXT_FRAMES"))
        if hasattr(module, "TRIGGER_MAX_BUFFER_FRAMES"):
            flags["max_buffer_frames"] = int(getattr(module, "TRIGGER_MAX_BUFFER_FRAMES"))
        if hasattr(module, "HAND_MASK_VALIDITY_SCALE"):
            flags["hand_mask_validity_scale"] = float(getattr(module, "HAND_MASK_VALIDITY_SCALE"))
        if hasattr(module, "POSE_HIP_COORDINATE_SCALE"):
            flags["pose_hip_coordinate_scale"] = float(getattr(module, "POSE_HIP_COORDINATE_SCALE"))
        if hasattr(module, "POSE_LOCAL_ANCHOR"):
            flags["pose_local_anchor"] = str(getattr(module, "POSE_LOCAL_ANCHOR")).strip().lower()
        if hasattr(module, "ENABLE_MOTHER_NOSE_Z_CALIBRATION"):
            flags["enable_mother_nose_z_calibration"] = _coerce_bool(
                getattr(module, "ENABLE_MOTHER_NOSE_Z_CALIBRATION")
            )
        if hasattr(module, "MOTHER_NOSE_Z_OFFSET"):
            flags["mother_nose_z_offset"] = float(getattr(module, "MOTHER_NOSE_Z_OFFSET"))
        if hasattr(module, "ENABLE_YOU_I_LEFTBODY_CALIBRATION"):
            flags["enable_you_i_leftbody_calibration"] = _coerce_bool(
                getattr(module, "ENABLE_YOU_I_LEFTBODY_CALIBRATION")
            )
        if hasattr(module, "YOU_I_LEFTBODY_Y_OFFSET"):
            flags["you_i_leftbody_y_offset"] = float(getattr(module, "YOU_I_LEFTBODY_Y_OFFSET"))
        if hasattr(module, "ENABLE_YOU_LIKE_PAIRWISE_CALIBRATION"):
            flags["enable_you_like_pairwise_calibration"] = _coerce_bool(
                getattr(module, "ENABLE_YOU_LIKE_PAIRWISE_CALIBRATION")
            )
        if hasattr(module, "YOU_LIKE_PAIRWISE_DELTA"):
            flags["you_like_pairwise_delta"] = float(getattr(module, "YOU_LIKE_PAIRWISE_DELTA"))
        if hasattr(module, "ENABLE_LIKE_I_PAIRWISE_CALIBRATION"):
            flags["enable_like_i_pairwise_calibration"] = _coerce_bool(
                getattr(module, "ENABLE_LIKE_I_PAIRWISE_CALIBRATION")
            )
        if hasattr(module, "LIKE_I_PAIRWISE_DELTA"):
            flags["like_i_pairwise_delta"] = float(getattr(module, "LIKE_I_PAIRWISE_DELTA"))
        if hasattr(module, "ENABLE_YOU_LIKE_FRONTLOADED_RESCORE"):
            flags["enable_you_like_frontloaded_rescore"] = _coerce_bool(
                getattr(module, "ENABLE_YOU_LIKE_FRONTLOADED_RESCORE")
            )
        if hasattr(module, "YOU_LIKE_FRONTLOAD_POWER"):
            flags["you_like_frontload_power"] = float(getattr(module, "YOU_LIKE_FRONTLOAD_POWER"))
        if hasattr(module, "ENABLE_WANT_ANCHOR_FALLBACK"):
            flags["enable_want_anchor_fallback"] = _coerce_bool(
                getattr(module, "ENABLE_WANT_ANCHOR_FALLBACK")
            )
        if hasattr(module, "WANT_ANCHOR_FALLBACK_MAX_NOSIGN_CONFIDENCE"):
            flags["want_anchor_fallback_max_nosign_confidence"] = float(
                getattr(module, "WANT_ANCHOR_FALLBACK_MAX_NOSIGN_CONFIDENCE")
            )
        if hasattr(module, "ENABLE_FATHER_TRIGGER_RESCUE"):
            flags["enable_father_trigger_rescue"] = _coerce_bool(
                getattr(module, "ENABLE_FATHER_TRIGGER_RESCUE")
            )
        if hasattr(module, "FATHER_TRIGGER_RESCUE_PAIRWISE_DELTA"):
            flags["father_trigger_rescue_pairwise_delta"] = float(
                getattr(module, "FATHER_TRIGGER_RESCUE_PAIRWISE_DELTA")
            )
        if hasattr(module, "FATHER_TRIGGER_RESCUE_MIN_RATIO"):
            flags["father_trigger_rescue_min_ratio"] = float(
                getattr(module, "FATHER_TRIGGER_RESCUE_MIN_RATIO")
            )
        if hasattr(module, "FATHER_TRIGGER_RESCUE_MAX_NOSIGN_CONFIDENCE"):
            flags["father_trigger_rescue_max_nosign_confidence"] = float(
                getattr(module, "FATHER_TRIGGER_RESCUE_MAX_NOSIGN_CONFIDENCE")
            )
    except Exception:
        return flags
    env_overrides = {
        "TRIGGER_MIN_ACTION_FRAMES": ("min_action_frames", int),
        "TRIGGER_IDLE_PATIENCE": ("idle_patience", int),
        "TRIGGER_MIN_MOTION_ENERGY": ("min_motion_energy", float),
        "TRIGGER_MIN_TOP_MARGIN": ("min_top_margin", float),
        "TRIGGER_PRE_CONTEXT_FRAMES": ("pre_context_frames", int),
        "TRIGGER_MAX_BUFFER_FRAMES": ("max_buffer_frames", int),
        "HAND_MASK_VALIDITY_SCALE": ("hand_mask_validity_scale", float),
        "POSE_HIP_COORDINATE_SCALE": ("pose_hip_coordinate_scale", float),
        "POSE_LOCAL_ANCHOR": ("pose_local_anchor", str),
        "ENABLE_MOTHER_NOSE_Z_CALIBRATION": ("enable_mother_nose_z_calibration", _coerce_bool),
        "MOTHER_NOSE_Z_OFFSET": ("mother_nose_z_offset", float),
        "ENABLE_YOU_I_LEFTBODY_CALIBRATION": ("enable_you_i_leftbody_calibration", _coerce_bool),
        "YOU_I_LEFTBODY_Y_OFFSET": ("you_i_leftbody_y_offset", float),
        "ENABLE_YOU_LIKE_PAIRWISE_CALIBRATION": ("enable_you_like_pairwise_calibration", _coerce_bool),
        "YOU_LIKE_PAIRWISE_DELTA": ("you_like_pairwise_delta", float),
        "ENABLE_LIKE_I_PAIRWISE_CALIBRATION": ("enable_like_i_pairwise_calibration", _coerce_bool),
        "LIKE_I_PAIRWISE_DELTA": ("like_i_pairwise_delta", float),
        "ENABLE_YOU_LIKE_FRONTLOADED_RESCORE": ("enable_you_like_frontloaded_rescore", _coerce_bool),
        "YOU_LIKE_FRONTLOAD_POWER": ("you_like_frontload_power", float),
        "ENABLE_WANT_ANCHOR_FALLBACK": ("enable_want_anchor_fallback", _coerce_bool),
        "WANT_ANCHOR_FALLBACK_MAX_NOSIGN_CONFIDENCE": ("want_anchor_fallback_max_nosign_confidence", float),
        "ENABLE_FATHER_TRIGGER_RESCUE": ("enable_father_trigger_rescue", _coerce_bool),
        "FATHER_TRIGGER_RESCUE_PAIRWISE_DELTA": ("father_trigger_rescue_pairwise_delta", float),
        "FATHER_TRIGGER_RESCUE_MIN_RATIO": ("father_trigger_rescue_min_ratio", float),
        "FATHER_TRIGGER_RESCUE_MAX_NOSIGN_CONFIDENCE": ("father_trigger_rescue_max_nosign_confidence", float),
    }
    for env_name, (flag_name, caster) in env_overrides.items():
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        try:
            flags[flag_name] = caster(raw_value)
        except (TypeError, ValueError):
            continue
    return flags


def load_gesture_assist_settings() -> dict[str, float]:
    settings = copy.deepcopy(DEFAULT_GESTURE_ASSIST_SETTINGS)
    if not USER_TUNING_SETTINGS_PATH.exists():
        return settings
    try:
        spec = importlib.util.spec_from_file_location("user_realtime_tuning_settings", USER_TUNING_SETTINGS_PATH)
        if spec is None or spec.loader is None:
            return settings
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        overrides = getattr(module, "GESTURE_ASSIST_SETTINGS", {})
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                if key in settings:
                    settings[key] = float(value)
    except Exception:
        return settings
    return settings


@dataclass
class FramePrediction:
    display_frame: np.ndarray
    decision: InferenceDecision
    engine_mode: str
    emitted_label: str
    checkpoint_branch: str
    frame_index: int
    top_candidates: list[tuple[str, float]]
    probabilities: dict[str, float]
    assisted_top_candidates: list[tuple[str, float]]
    assisted_probabilities: dict[str, float]
    assist_notes: str
    disambiguation_notes: str
    disambiguation_applied: bool
    left_hand_present: bool
    right_hand_present: bool
    pose_present: bool
    motion_energy: float
    top_margin: float
    signal_score: float
    gesture_features: dict[str, float]
    profile_probabilities: dict[str, float]
    profile_top_candidates: list[tuple[str, float]]
    trigger_segment_debug: dict[str, object] | None = None


@dataclass
class TokenEvent:
    label: str
    start_frame: int
    end_frame: int
    confidence: float


@dataclass
class TriggerFrameRecord:
    frame_index: int
    frame_landmarks: FrameLandmarks
    motion_energy: float
    left_hand_present: bool
    right_hand_present: bool
    pose_present: bool


@dataclass
class TriggerSegmentDebug:
    segment_id: int
    raw_start_frame: int
    raw_end_frame: int
    start_frame: int
    end_frame: int
    decision_frame: int
    raw_length: int
    trimmed_length: int
    pre_context_frames: int
    tail_trimmed: int
    tail_trimmed_no_hand: int
    end_reason: str
    trigger_end_reason: str
    raw_frame_indices: list[int] = field(default_factory=list)
    sampled_indices: list[int] = field(default_factory=list)
    sampled_frame_indices: list[int] = field(default_factory=list)
    motion_energy_curve: list[float] = field(default_factory=list)
    raw_label: str = ""
    raw_confidence: float = 0.0
    top_margin: float = 0.0
    emitted_label: str = ""
    decision_status: str = ""
    top_candidates: list[dict[str, float | str]] = field(default_factory=list)
    top_logits: list[dict[str, float | str]] = field(default_factory=list)
    sampled_path_top_candidates: list[dict[str, float | str]] = field(default_factory=list)
    sampled_path_top_logits: list[dict[str, float | str]] = field(default_factory=list)
    sampled_path_raw_label: str = ""
    sampled_path_raw_confidence: float = 0.0
    feature_path_l2_mean: float = 0.0
    feature_path_l2_max: float = 0.0
    feature_path_cosine_mean: float = 0.0
    segment_has_left_hand: bool = False
    segment_has_right_hand: bool = False
    segment_has_both_hands: bool = False
    segment_pose_ratio: float = 0.0
    you_like_frontload_attempted: bool = False
    you_like_frontload_applied: bool = False
    you_like_frontload_power: float = 0.0
    you_like_frontload_top_margin: float = 0.0
    you_like_frontload_raw_confidence: float = 0.0
    you_like_frontload_top_candidates: list[dict[str, float | str]] = field(default_factory=list)
    you_like_frontload_sampled_indices: list[int] = field(default_factory=list)
    you_like_frontload_sampled_frame_indices: list[int] = field(default_factory=list)
    you_like_pairwise_attempted: bool = False
    you_like_pairwise_applied: bool = False
    you_like_pairwise_delta: float = 0.0
    you_like_pairwise_top_margin: float = 0.0
    you_like_pairwise_raw_confidence: float = 0.0
    you_like_pairwise_top_candidates: list[dict[str, float | str]] = field(default_factory=list)
    like_i_pairwise_attempted: bool = False
    like_i_pairwise_applied: bool = False
    like_i_pairwise_delta: float = 0.0
    like_i_pairwise_top_margin: float = 0.0
    like_i_pairwise_raw_confidence: float = 0.0
    like_i_pairwise_top_candidates: list[dict[str, float | str]] = field(default_factory=list)
    want_anchor_fallback_attempted: bool = False
    want_anchor_fallback_applied: bool = False
    want_anchor_fallback_raw_label: str = ""
    want_anchor_fallback_raw_confidence: float = 0.0
    want_anchor_fallback_top_margin: float = 0.0
    want_anchor_fallback_top_candidates: list[dict[str, float | str]] = field(default_factory=list)
    want_anchor_fallback_candidate_anchor: str = ""
    father_trigger_rescue_attempted: bool = False
    father_trigger_rescue_applied: bool = False
    father_trigger_rescue_hit_ratio: float = 0.0
    father_trigger_rescue_valid_count: int = 0
    father_trigger_rescue_pairwise_delta: float = 0.0
    father_trigger_rescue_raw_label: str = ""
    father_trigger_rescue_raw_confidence: float = 0.0
    father_trigger_rescue_top_margin: float = 0.0
    father_trigger_rescue_top_candidates: list[dict[str, float | str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_id": int(self.segment_id),
            "raw_start_frame": int(self.raw_start_frame),
            "raw_end_frame": int(self.raw_end_frame),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "decision_frame": int(self.decision_frame),
            "raw_length": int(self.raw_length),
            "trimmed_length": int(self.trimmed_length),
            "pre_context_frames": int(self.pre_context_frames),
            "tail_trimmed": int(self.tail_trimmed),
            "tail_trimmed_no_hand": int(self.tail_trimmed_no_hand),
            "end_reason": self.end_reason,
            "trigger_end_reason": self.trigger_end_reason,
            "raw_frame_indices": list(self.raw_frame_indices),
            "sampled_indices": list(self.sampled_indices),
            "sampled_frame_indices": list(self.sampled_frame_indices),
            "motion_energy_curve": [float(value) for value in self.motion_energy_curve],
            "raw_label": self.raw_label,
            "raw_confidence": float(self.raw_confidence),
            "top_margin": float(self.top_margin),
            "emitted_label": self.emitted_label,
            "decision_status": self.decision_status,
            "top_candidates": list(self.top_candidates),
            "top_logits": list(self.top_logits),
            "sampled_path_top_candidates": list(self.sampled_path_top_candidates),
            "sampled_path_top_logits": list(self.sampled_path_top_logits),
            "sampled_path_raw_label": self.sampled_path_raw_label,
            "sampled_path_raw_confidence": float(self.sampled_path_raw_confidence),
            "feature_path_l2_mean": float(self.feature_path_l2_mean),
            "feature_path_l2_max": float(self.feature_path_l2_max),
            "feature_path_cosine_mean": float(self.feature_path_cosine_mean),
            "segment_has_left_hand": bool(self.segment_has_left_hand),
            "segment_has_right_hand": bool(self.segment_has_right_hand),
            "segment_has_both_hands": bool(self.segment_has_both_hands),
            "segment_pose_ratio": float(self.segment_pose_ratio),
            "you_like_frontload_attempted": bool(self.you_like_frontload_attempted),
            "you_like_frontload_applied": bool(self.you_like_frontload_applied),
            "you_like_frontload_power": float(self.you_like_frontload_power),
            "you_like_frontload_top_margin": float(self.you_like_frontload_top_margin),
            "you_like_frontload_raw_confidence": float(self.you_like_frontload_raw_confidence),
            "you_like_frontload_top_candidates": list(self.you_like_frontload_top_candidates),
            "you_like_frontload_sampled_indices": list(self.you_like_frontload_sampled_indices),
            "you_like_frontload_sampled_frame_indices": list(self.you_like_frontload_sampled_frame_indices),
            "you_like_pairwise_attempted": bool(self.you_like_pairwise_attempted),
            "you_like_pairwise_applied": bool(self.you_like_pairwise_applied),
            "you_like_pairwise_delta": float(self.you_like_pairwise_delta),
            "you_like_pairwise_top_margin": float(self.you_like_pairwise_top_margin),
            "you_like_pairwise_raw_confidence": float(self.you_like_pairwise_raw_confidence),
            "you_like_pairwise_top_candidates": list(self.you_like_pairwise_top_candidates),
            "like_i_pairwise_attempted": bool(self.like_i_pairwise_attempted),
            "like_i_pairwise_applied": bool(self.like_i_pairwise_applied),
            "like_i_pairwise_delta": float(self.like_i_pairwise_delta),
            "like_i_pairwise_top_margin": float(self.like_i_pairwise_top_margin),
            "like_i_pairwise_raw_confidence": float(self.like_i_pairwise_raw_confidence),
            "like_i_pairwise_top_candidates": list(self.like_i_pairwise_top_candidates),
            "want_anchor_fallback_attempted": bool(self.want_anchor_fallback_attempted),
            "want_anchor_fallback_applied": bool(self.want_anchor_fallback_applied),
            "want_anchor_fallback_raw_label": self.want_anchor_fallback_raw_label,
            "want_anchor_fallback_raw_confidence": float(self.want_anchor_fallback_raw_confidence),
            "want_anchor_fallback_top_margin": float(self.want_anchor_fallback_top_margin),
            "want_anchor_fallback_top_candidates": list(self.want_anchor_fallback_top_candidates),
            "want_anchor_fallback_candidate_anchor": self.want_anchor_fallback_candidate_anchor,
            "father_trigger_rescue_attempted": bool(self.father_trigger_rescue_attempted),
            "father_trigger_rescue_applied": bool(self.father_trigger_rescue_applied),
            "father_trigger_rescue_hit_ratio": float(self.father_trigger_rescue_hit_ratio),
            "father_trigger_rescue_valid_count": int(self.father_trigger_rescue_valid_count),
            "father_trigger_rescue_pairwise_delta": float(self.father_trigger_rescue_pairwise_delta),
            "father_trigger_rescue_raw_label": self.father_trigger_rescue_raw_label,
            "father_trigger_rescue_raw_confidence": float(self.father_trigger_rescue_raw_confidence),
            "father_trigger_rescue_top_margin": float(self.father_trigger_rescue_top_margin),
            "father_trigger_rescue_top_candidates": list(self.father_trigger_rescue_top_candidates),
        }


class PhraseCollector:
    def __init__(
        self,
        min_gap_frames: int = 24,
        stable_frames: int = 3,
        release_frames: int = 6,
        min_token_confidence: float = 0.72,
        switch_frames: int = 5,
        min_segment_frames: int = 6,
    ) -> None:
        self.min_gap_frames = min_gap_frames
        self.stable_frames = stable_frames
        self.release_frames = release_frames
        self.min_token_confidence = min_token_confidence
        self.switch_frames = switch_frames
        self.min_segment_frames = min_segment_frames
        self.tokens: list[str] = []
        self._last_label = ""
        self._last_frame_index = -10_000
        self._candidate_label = ""
        self._candidate_count = 0
        self._active_label = ""
        self._silence_count = 0
        self._segment_active = False
        self._segment_start_frame = -1
        self._segment_scores: dict[str, float] = {}
        self._segment_counts: dict[str, int] = {}
        self._segment_peak_confidence = 0.0
        self._switch_candidate = ""
        self._switch_count = 0

    def clear(self) -> None:
        self.tokens.clear()
        self._last_label = ""
        self._last_frame_index = -10_000
        self._candidate_label = ""
        self._candidate_count = 0
        self._active_label = ""
        self._silence_count = 0
        self._segment_active = False
        self._segment_start_frame = -1
        self._segment_scores = {}
        self._segment_counts = {}
        self._segment_peak_confidence = 0.0
        self._switch_candidate = ""
        self._switch_count = 0

    def update(self, label: str, frame_index: int, confidence: float = 0.0) -> TokenEvent | None:
        normalized = label.strip().lower()
        if not normalized or normalized in IGNORED_LABELS:
            self._candidate_label = ""
            self._candidate_count = 0
            self._silence_count += 1
            if self._segment_active and self._silence_count >= self.release_frames:
                return self._finalize_segment(frame_index - self.release_frames)
            if self._silence_count >= self.release_frames:
                self._active_label = ""
            return None
        self._silence_count = 0

        if not self._segment_active:
            if confidence < self.min_token_confidence:
                self._candidate_label = ""
                self._candidate_count = 0
                return None
            if normalized == self._candidate_label:
                self._candidate_count += 1
            else:
                self._candidate_label = normalized
                self._candidate_count = 1
            if self._candidate_count < self.stable_frames:
                return None
            self._start_segment(normalized, frame_index, confidence)
            return None

        self._consume_active_frame(normalized, confidence)
        current_best_label = self._best_segment_label()
        if normalized != current_best_label and confidence >= self.min_token_confidence:
            if normalized == self._switch_candidate:
                self._switch_count += 1
            else:
                self._switch_candidate = normalized
                self._switch_count = 1
            if self._switch_count >= self.switch_frames:
                event = self._finalize_segment(frame_index - self.switch_frames)
                self._start_segment(normalized, frame_index, confidence)
                return event
            return None

        self._switch_candidate = ""
        self._switch_count = 0
        return None

    def flush(self, frame_index: int) -> TokenEvent | None:
        if not self._segment_active:
            return None
        return self._finalize_segment(frame_index)

    def _start_segment(self, label: str, frame_index: int, confidence: float) -> None:
        self._segment_active = True
        self._segment_start_frame = max(0, frame_index - self.stable_frames + 1)
        self._segment_scores = {label: confidence * self.stable_frames}
        self._segment_counts = {label: self.stable_frames}
        self._segment_peak_confidence = confidence
        self._active_label = label
        self._candidate_label = ""
        self._candidate_count = 0
        self._switch_candidate = ""
        self._switch_count = 0

    def _consume_active_frame(self, label: str, confidence: float) -> None:
        self._segment_scores[label] = self._segment_scores.get(label, 0.0) + confidence
        self._segment_counts[label] = self._segment_counts.get(label, 0) + 1
        self._segment_peak_confidence = max(self._segment_peak_confidence, confidence)

    def _best_segment_label(self) -> str:
        best_label = ""
        best_score = -1.0
        for label, score in self._segment_scores.items():
            count_bonus = 0.05 * self._segment_counts.get(label, 0)
            final_score = score + count_bonus
            if final_score > best_score:
                best_score = final_score
                best_label = label
        return best_label

    def _finalize_segment(self, end_frame: int) -> TokenEvent | None:
        if not self._segment_active:
            return None

        label = self._best_segment_label()
        segment_length = max(0, end_frame - self._segment_start_frame + 1)
        average_confidence = self._segment_scores.get(label, 0.0) / max(self._segment_counts.get(label, 1), 1)

        self._segment_active = False
        self._active_label = ""
        self._segment_scores = {}
        self._segment_counts = {}
        self._segment_peak_confidence = 0.0
        self._switch_candidate = ""
        self._switch_count = 0

        if segment_length < self.min_segment_frames:
            return None
        if average_confidence < self.min_token_confidence and self._last_label:
            return None
        if label == self._last_label and self._segment_start_frame - self._last_frame_index < self.min_gap_frames:
            return None

        self.tokens.append(label)
        self._last_label = label
        self._last_frame_index = end_frame
        return TokenEvent(
            label=label,
            start_frame=self._segment_start_frame,
            end_frame=end_frame,
            confidence=average_confidence,
        )


class MultibranchSequenceEngine:
    def __init__(
        self,
        checkpoint_path: str | Path,
        sequence_length: int,
        confidence_threshold: float,
        gesture_profile_path: str | Path | None = None,
        mode: str = "sliding_window",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        if mode not in {"sliding_window", "trigger_based"}:
            raise ValueError(f"Unsupported mode={mode}. Expected 'sliding_window' or 'trigger_based'.")
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.checkpoint = load_multibranch_checkpoint(self.checkpoint_path, self.device)
        if self.model is None or self.checkpoint is None:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint_mode = self.checkpoint.get("feature_mode")
        checkpoint_dim = self.checkpoint.get("input_dim")
        if checkpoint_mode != CONFIG.feature_mode:
            raise RuntimeError(
                f"Checkpoint feature_mode={checkpoint_mode} does not match current config feature_mode={CONFIG.feature_mode}"
            )
        self.feature_spec = resolve_feature_spec(
            feature_mode=checkpoint_mode,
            feature_spec=self.checkpoint.get("feature_spec"),
            total_dim=checkpoint_dim,
        )

        self.index_to_label = {int(index): label for index, label in self.checkpoint["index_to_label"].items()}
        self.checkpoint_branch = self.checkpoint.get("skeleton_branch_type")
        if self.checkpoint_branch is None:
            self.checkpoint_branch = "gcn" if self.checkpoint.get("use_gcn_skeleton", False) else "lstm"

        self.extractor = HolisticExtractor()
        self.postprocessor = InferencePostprocessor()
        self.builder = SequenceBuilder(sequence_length)
        self.frame_valid_builder = SequenceBuilder(sequence_length)
        self.prev_feature = None
        self.frame_index = 0
        self.target_frame_size: tuple[int, int] | None = None
        self.gesture_profiles = self._load_gesture_profiles(gesture_profile_path)
        self.gesture_assist_settings = load_gesture_assist_settings()
        self.runtime_inference_flags = load_runtime_inference_flags()
        self.enable_rule_based_disambiguation = bool(
            self.runtime_inference_flags["enable_rule_based_disambiguation"]
        )
        self.enable_dynamic_nosign_suppression = bool(
            self.runtime_inference_flags["enable_dynamic_nosign_suppression"]
        )
        self.nosign_penalty_factor = float(self.runtime_inference_flags["nosign_penalty_factor"])
        self.trigger_motion_threshold = max(0.0, float(self.runtime_inference_flags["min_motion_energy"]))
        self.trigger_idle_threshold = 0.008
        self.trigger_start_frames = max(1, int(self.runtime_inference_flags["trigger_patience"]))
        self.trigger_end_frames = max(1, int(self.runtime_inference_flags["idle_patience"]))
        self.trigger_min_action_frames = max(1, int(self.runtime_inference_flags["min_action_frames"]))
        self.trigger_emit_confidence_threshold = float(self.runtime_inference_flags["emit_confidence_threshold"])
        self.trigger_min_top_margin = float(self.runtime_inference_flags["min_top_margin"])
        self.trigger_pre_context_frames = max(0, int(self.runtime_inference_flags["pre_context_frames"]))
        self.trigger_max_buffer_frames = max(int(self.runtime_inference_flags["max_buffer_frames"]), self.sequence_length)
        self.hand_mask_validity_scale = float(self.runtime_inference_flags["hand_mask_validity_scale"])
        self.pose_hip_coordinate_scale = float(self.runtime_inference_flags["pose_hip_coordinate_scale"])
        self.pose_local_anchor = str(self.runtime_inference_flags["pose_local_anchor"]).strip().lower()
        self.enable_mother_nose_z_calibration = bool(
            self.runtime_inference_flags["enable_mother_nose_z_calibration"]
        )
        self.mother_nose_z_offset = float(self.runtime_inference_flags["mother_nose_z_offset"])
        self.enable_you_i_leftbody_calibration = bool(
            self.runtime_inference_flags["enable_you_i_leftbody_calibration"]
        )
        self.you_i_leftbody_y_offset = float(self.runtime_inference_flags["you_i_leftbody_y_offset"])
        self.enable_you_like_pairwise_calibration = bool(
            self.runtime_inference_flags["enable_you_like_pairwise_calibration"]
        )
        self.you_like_pairwise_delta = float(self.runtime_inference_flags["you_like_pairwise_delta"])
        self.enable_like_i_pairwise_calibration = bool(
            self.runtime_inference_flags["enable_like_i_pairwise_calibration"]
        )
        self.like_i_pairwise_delta = float(self.runtime_inference_flags["like_i_pairwise_delta"])
        self.enable_you_like_frontloaded_rescore = bool(
            self.runtime_inference_flags["enable_you_like_frontloaded_rescore"]
        )
        self.you_like_frontload_power = float(self.runtime_inference_flags["you_like_frontload_power"])
        self.enable_want_anchor_fallback = bool(
            self.runtime_inference_flags["enable_want_anchor_fallback"]
        )
        self.want_anchor_fallback_max_nosign_confidence = float(
            self.runtime_inference_flags["want_anchor_fallback_max_nosign_confidence"]
        )
        self.enable_father_trigger_rescue = bool(
            self.runtime_inference_flags["enable_father_trigger_rescue"]
        )
        self.father_trigger_rescue_pairwise_delta = float(
            self.runtime_inference_flags["father_trigger_rescue_pairwise_delta"]
        )
        self.father_trigger_rescue_min_ratio = float(
            self.runtime_inference_flags["father_trigger_rescue_min_ratio"]
        )
        self.father_trigger_rescue_max_nosign_confidence = float(
            self.runtime_inference_flags["father_trigger_rescue_max_nosign_confidence"]
        )
        self._last_you_like_pairwise_debug = {
            "attempted": False,
            "applied": False,
            "delta": 0.0,
            "raw_confidence": 0.0,
            "top_margin": 0.0,
            "top_candidates": [],
        }
        self._last_like_i_pairwise_debug = {
            "attempted": False,
            "applied": False,
            "delta": 0.0,
            "raw_confidence": 0.0,
            "top_margin": 0.0,
            "top_candidates": [],
        }
        self._last_want_anchor_fallback_debug = {
            "attempted": False,
            "applied": False,
            "raw_label": "",
            "raw_confidence": 0.0,
            "top_margin": 0.0,
            "top_candidates": [],
            "candidate_anchor": "",
        }
        self._last_father_trigger_rescue_debug = {
            "attempted": False,
            "applied": False,
            "hit_ratio": 0.0,
            "valid_count": 0,
            "pairwise_delta": 0.0,
            "raw_label": "",
            "raw_confidence": 0.0,
            "top_margin": 0.0,
            "top_candidates": [],
        }
        self.trigger_pre_buffer: deque[TriggerFrameRecord] = deque(maxlen=self.trigger_pre_context_frames)
        self.trigger_action_buffer: list[TriggerFrameRecord] = []
        self.trigger_start_count = 0
        self.trigger_idle_count = 0
        self.trigger_active = False
        self.trigger_segment_id = 0
        self.trigger_current_pre_context_count = 0
        self.disambiguator = ClassSpecificDisambiguator()
        self.mother_you_left_face_prototypes = self._load_mother_you_left_face_prototypes()
        self.you_i_left_body_prototypes = self._load_you_i_left_body_prototypes()
        self.like_i_location_prototypes = self._load_like_i_location_prototypes()

    def _motion_energy(self, built: dict[str, np.ndarray]) -> float:
        motion = built.get("motion_components", {})
        left_velocity = np.asarray(motion.get("left_velocity", np.zeros((3,), dtype=np.float32)), dtype=np.float32)
        right_velocity = np.asarray(motion.get("right_velocity", np.zeros((3,), dtype=np.float32)), dtype=np.float32)
        distance_delta = np.asarray(motion.get("distance_delta", np.zeros((1,), dtype=np.float32)), dtype=np.float32)
        energy = float(np.linalg.norm(left_velocity) + np.linalg.norm(right_velocity) + np.abs(distance_delta).sum())
        return energy

    def _signal_score(self, *, valid_ratio: float, left_hand_present: bool, right_hand_present: bool, pose_present: bool, motion_energy: float) -> float:
        hand_score = (float(left_hand_present) + float(right_hand_present)) / 2.0
        pose_score = 1.0 if pose_present else 0.0
        motion_score = min(motion_energy / 0.08, 1.0)
        score = 0.45 * valid_ratio + 0.25 * hand_score + 0.10 * pose_score + 0.20 * motion_score
        return float(min(1.0, max(0.0, score)))

    def _top_candidates_from_probabilities(self, probabilities: dict[str, float], limit: int = 3) -> list[tuple[str, float]]:
        return sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:limit]

    def _top_logits_from_scores(self, logits: dict[str, float], limit: int = 3) -> list[tuple[str, float]]:
        return sorted(logits.items(), key=lambda item: item[1], reverse=True)[:limit]

    def _serialize_ranked_scores(
        self,
        ranked_scores: list[tuple[str, float]],
        *,
        key_name: str,
        limit: int = 5,
    ) -> list[dict[str, float | str]]:
        return [
            {"label": label, key_name: float(score)}
            for label, score in ranked_scores[:limit]
        ]

    def _copy_frame_landmarks(self, frame_landmarks: FrameLandmarks) -> FrameLandmarks:
        return FrameLandmarks(
            left_hand=frame_landmarks.left_hand.astype(np.float32, copy=True),
            right_hand=frame_landmarks.right_hand.astype(np.float32, copy=True),
            pose=frame_landmarks.pose.astype(np.float32, copy=True),
            mouth_center=frame_landmarks.mouth_center.astype(np.float32, copy=True),
            chin=frame_landmarks.chin.astype(np.float32, copy=True),
            left_hand_mask=frame_landmarks.left_hand_mask.astype(np.float32, copy=True),
            right_hand_mask=frame_landmarks.right_hand_mask.astype(np.float32, copy=True),
            pose_mask=frame_landmarks.pose_mask.astype(np.float32, copy=True),
            mouth_mask=frame_landmarks.mouth_mask.astype(np.float32, copy=True),
            chin_mask=frame_landmarks.chin_mask.astype(np.float32, copy=True),
        )

    def _make_trigger_frame_record(
        self,
        *,
        frame_index: int,
        frame_landmarks: FrameLandmarks,
        motion_energy: float,
        left_hand_present: bool,
        right_hand_present: bool,
        pose_present: bool,
    ) -> TriggerFrameRecord:
        return TriggerFrameRecord(
            frame_index=frame_index,
            frame_landmarks=self._copy_frame_landmarks(frame_landmarks),
            motion_energy=float(motion_energy),
            left_hand_present=bool(left_hand_present),
            right_hand_present=bool(right_hand_present),
            pose_present=bool(pose_present),
        )

    def _build_feature_rows_from_records(
        self,
        frames: list[TriggerFrameRecord],
        *,
        pose_local_anchor: str | None = None,
    ) -> list[np.ndarray]:
        feature_rows: list[np.ndarray] = []
        prev_feature = None
        anchor = str(pose_local_anchor or self.pose_local_anchor).strip().lower()
        for record in frames:
            frame_landmarks = record.frame_landmarks
            built = build_frame_feature(
                left_hand=frame_landmarks.left_hand,
                right_hand=frame_landmarks.right_hand,
                pose=frame_landmarks.pose,
                mouth_center=frame_landmarks.mouth_center,
                chin=frame_landmarks.chin,
                left_hand_mask=frame_landmarks.left_hand_mask,
                right_hand_mask=frame_landmarks.right_hand_mask,
                pose_mask=frame_landmarks.pose_mask,
                mouth_mask=frame_landmarks.mouth_mask,
                chin_mask=frame_landmarks.chin_mask,
                prev_feature=prev_feature,
                feature_mode=CONFIG.feature_mode,
                feature_spec=self.feature_spec,
                pose_local_anchor=anchor,
            )
            prev_feature = built
            feature_rows.append(built["feature_vector"])
        return feature_rows

    def _materialize_trigger_sequence(
        self,
        *,
        frames: list[TriggerFrameRecord],
        feature_rows: list[np.ndarray],
        sampled_index_list: list[int],
        pose_local_anchor: str | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        sequence_rows = [feature_rows[index] for index in sampled_index_list]
        sampled_frames = [frames[index] for index in sampled_index_list]
        sampled_path_rows = self._build_feature_rows_from_records(
            sampled_frames,
            pose_local_anchor=pose_local_anchor,
        )
        sampled_path_sequence = np.stack(sampled_path_rows, axis=0).astype(np.float32)
        contiguous_sampled_sequence = np.stack(sequence_rows, axis=0).astype(np.float32)
        l2_curve = np.linalg.norm(contiguous_sampled_sequence - sampled_path_sequence, axis=1)
        cosine_curve = []
        for left_row, right_row in zip(contiguous_sampled_sequence, sampled_path_sequence):
            left_norm = float(np.linalg.norm(left_row))
            right_norm = float(np.linalg.norm(right_row))
            if left_norm <= 1e-8 or right_norm <= 1e-8:
                cosine_curve.append(1.0)
                continue
            cosine_curve.append(float(np.dot(left_row, right_row) / (left_norm * right_norm)))
        return contiguous_sampled_sequence, {
            "sampled_indices": list(sampled_index_list),
            "sampled_frame_indices": [int(record.frame_index) for record in sampled_frames],
            "sampled_path_sequence": sampled_path_sequence,
            "feature_path_l2_mean": float(np.mean(l2_curve)) if len(l2_curve) else 0.0,
            "feature_path_l2_max": float(np.max(l2_curve)) if len(l2_curve) else 0.0,
            "feature_path_cosine_mean": float(np.mean(cosine_curve)) if cosine_curve else 0.0,
        }

    def _frontloaded_trigger_sample_indices(self, frame_count: int, *, power: float) -> list[int]:
        if frame_count <= 0:
            return []
        positions = np.linspace(0.0, 1.0, self.sequence_length, dtype=np.float64)
        sampled = np.rint(((positions**max(power, 1.0)) * max(frame_count - 1, 0))).astype(int)
        sampled = np.clip(sampled, 0, max(frame_count - 1, 0))
        return [int(index) for index in sampled.tolist()]

    def _build_trigger_sequence(
        self,
        frames: list[TriggerFrameRecord],
        *,
        pose_local_anchor: str | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if not frames:
            raise ValueError("Expected at least one raw landmark frame for trigger sequence.")
        anchor = str(pose_local_anchor or self.pose_local_anchor).strip().lower()
        feature_rows = self._build_feature_rows_from_records(frames, pose_local_anchor=anchor)
        sampled_index_list = [int(index) for index in np.linspace(0, len(frames) - 1, self.sequence_length).astype(int).tolist()]
        contiguous_sampled_sequence, sampled_debug = self._materialize_trigger_sequence(
            frames=frames,
            feature_rows=feature_rows,
            sampled_index_list=sampled_index_list,
            pose_local_anchor=anchor,
        )
        debug = {
            "raw_frame_indices": [int(record.frame_index) for record in frames],
            "motion_energy_curve": [float(record.motion_energy) for record in frames],
            "segment_has_left_hand": any(record.left_hand_present for record in frames),
            "segment_has_right_hand": any(record.right_hand_present for record in frames),
            "segment_has_both_hands": any(
                record.left_hand_present and record.right_hand_present
                for record in frames
            ),
            "segment_pose_ratio": (
                sum(float(record.pose_present) for record in frames) / max(len(frames), 1)
            ),
            "feature_rows": feature_rows,
            "pose_local_anchor": anchor,
        }
        debug.update(sampled_debug)
        return contiguous_sampled_sequence, debug

    def _left_face_feature_slice(self) -> slice | None:
        location_component = self.feature_spec.get("components", {}).get("location")
        if not isinstance(location_component, dict):
            return None
        start = int(location_component.get("start", -1))
        end = int(location_component.get("end", -1))
        if start < 0 or end < start + LEFT_FACE_VECTOR_DIM:
            return None
        return slice(start, start + LEFT_FACE_VECTOR_DIM)

    def _left_body_feature_slice(self) -> slice | None:
        location_component = self.feature_spec.get("components", {}).get("location")
        if not isinstance(location_component, dict):
            return None
        start = int(location_component.get("start", -1)) + LEFT_FACE_VECTOR_DIM
        end = int(location_component.get("end", -1))
        if start < 0 or end < start + LEFT_BODY_VECTOR_DIM:
            return None
        return slice(start, start + LEFT_BODY_VECTOR_DIM)

    def _location_feature_slice(self) -> slice | None:
        location_component = self.feature_spec.get("components", {}).get("location")
        if not isinstance(location_component, dict):
            return None
        start = int(location_component.get("start", -1))
        end = int(location_component.get("end", -1))
        if start < 0 or end <= start:
            return None
        return slice(start, end)

    def _you_i_leftbody_gap(self, adjusted_sequence: np.ndarray) -> tuple[float, float] | None:
        if {"you", "i"} - set(self.you_i_left_body_prototypes):
            return None
        left_body_slice = self._left_body_feature_slice()
        if left_body_slice is None:
            return None
        left_body_sequence = adjusted_sequence[:, left_body_slice].astype(np.float32, copy=False)
        if left_body_sequence.ndim != 2 or left_body_sequence.shape[1] != LEFT_BODY_VECTOR_DIM:
            return None
        left_body_mean = left_body_sequence.mean(axis=0)
        you_proto = self.you_i_left_body_prototypes["you"]
        i_proto = self.you_i_left_body_prototypes["i"]
        baseline_gap = float(np.linalg.norm(left_body_mean - you_proto) - np.linalg.norm(left_body_mean - i_proto))
        adjusted_mean = left_body_mean.copy()
        adjusted_mean[LEFT_BODY_CHEST_Y_INDEX] += float(self.you_i_leftbody_y_offset)
        adjusted_mean[LEFT_BODY_TORSO_Y_INDEX] += float(self.you_i_leftbody_y_offset)
        adjusted_gap = float(np.linalg.norm(adjusted_mean - you_proto) - np.linalg.norm(adjusted_mean - i_proto))
        return baseline_gap, adjusted_gap

    def _like_i_location_gap(self, adjusted_sequence: np.ndarray) -> float | None:
        if {"like", "i"} - set(self.like_i_location_prototypes):
            return None
        location_slice = self._location_feature_slice()
        if location_slice is None:
            return None
        location_sequence = adjusted_sequence[:, location_slice].astype(np.float32, copy=False)
        if location_sequence.ndim != 2:
            return None
        location_mean = location_sequence.mean(axis=0)
        like_proto = self.like_i_location_prototypes["like"]
        i_proto = self.like_i_location_prototypes["i"]
        return float(np.linalg.norm(location_mean - like_proto) - np.linalg.norm(location_mean - i_proto))

    def _read_manifest_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _prototype_dataset_roots(self) -> list[Path]:
        roots: list[Path] = []
        artifacts_root = self.checkpoint_path.parent.parent
        artifacts_name = artifacts_root.name
        if artifacts_name.startswith("artifacts_"):
            roots.append(CONFIG.project_root / artifacts_name.replace("artifacts_", "dataset_pipeline_", 1))
        roots.append(CONFIG.dataset_pipeline_root)
        roots.append(CONFIG.project_root / "dataset_pipeline_webcam9_relative_coord_v1")
        unique_roots: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            resolved = str(root.resolve()) if root.exists() else str(root)
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_roots.append(root)
        return unique_roots

    def _load_mother_you_left_face_prototypes(self) -> dict[str, np.ndarray]:
        if not self.enable_mother_nose_z_calibration:
            return {}
        left_face_slice = self._left_face_feature_slice()
        if left_face_slice is None:
            return {}
        manifest_path = None
        processed_sequences_dir = None
        for dataset_root in self._prototype_dataset_roots():
            candidate_manifest = dataset_root / "manifests" / "sequence_manifest.csv"
            candidate_sequences = dataset_root / "processed_sequences"
            if candidate_manifest.exists() and candidate_sequences.exists():
                manifest_path = candidate_manifest
                processed_sequences_dir = candidate_sequences
                break
        if manifest_path is None or processed_sequences_dir is None:
            return {}
        try:
            manifest_rows = self._read_manifest_rows(manifest_path)
        except OSError:
            return {}
        chosen_paths: dict[str, list[Path]] = defaultdict(list)
        seen_groups: dict[str, set[str]] = defaultdict(set)
        for row in manifest_rows:
            label = str(row.get("english_label", "")).strip().lower()
            if label not in {"mother", "you"}:
                continue
            if len(chosen_paths[label]) >= MOTHER_YOU_PROTOTYPE_GROUPS:
                continue
            group_key = str(row.get("source_group_key", "")).strip()
            if group_key and group_key in seen_groups[label]:
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            sample_path = processed_sequences_dir / f"{sample_id}.npz"
            if not sample_path.exists():
                continue
            if group_key:
                seen_groups[label].add(group_key)
            chosen_paths[label].append(sample_path)
        prototypes: dict[str, np.ndarray] = {}
        for label, paths in chosen_paths.items():
            vectors: list[np.ndarray] = []
            for path in paths:
                try:
                    with np.load(path, allow_pickle=False) as payload:
                        sequence = payload["sequence"].astype(np.float32)[:, left_face_slice]
                except (OSError, KeyError, ValueError):
                    continue
                vectors.append(sequence.mean(axis=0))
            if vectors:
                prototypes[label] = np.stack(vectors, axis=0).mean(axis=0)
        if {"mother", "you"} - set(prototypes):
            return {}
        return prototypes

    def _load_you_i_left_body_prototypes(self) -> dict[str, np.ndarray]:
        if not self.enable_you_i_leftbody_calibration:
            return {}
        left_body_slice = self._left_body_feature_slice()
        if left_body_slice is None:
            return {}
        manifest_path = None
        processed_sequences_dir = None
        for dataset_root in self._prototype_dataset_roots():
            candidate_manifest = dataset_root / "manifests" / "sequence_manifest.csv"
            candidate_sequences = dataset_root / "processed_sequences"
            if candidate_manifest.exists() and candidate_sequences.exists():
                manifest_path = candidate_manifest
                processed_sequences_dir = candidate_sequences
                break
        if manifest_path is None or processed_sequences_dir is None:
            return {}
        try:
            manifest_rows = self._read_manifest_rows(manifest_path)
        except OSError:
            return {}
        chosen_paths: dict[str, list[Path]] = defaultdict(list)
        seen_groups: dict[str, set[str]] = defaultdict(set)
        for row in manifest_rows:
            label = str(row.get("english_label", "")).strip().lower()
            if label not in {"you", "i"}:
                continue
            if len(chosen_paths[label]) >= YOU_I_LEFT_BODY_PROTOTYPE_GROUPS:
                continue
            group_key = str(row.get("source_group_key", "")).strip()
            if group_key and group_key in seen_groups[label]:
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            sample_path = processed_sequences_dir / f"{sample_id}.npz"
            if not sample_path.exists():
                continue
            if group_key:
                seen_groups[label].add(group_key)
            chosen_paths[label].append(sample_path)
        prototypes: dict[str, np.ndarray] = {}
        for label, paths in chosen_paths.items():
            vectors: list[np.ndarray] = []
            for path in paths:
                try:
                    with np.load(path, allow_pickle=False) as payload:
                        sequence = payload["sequence"].astype(np.float32)[:, left_body_slice]
                except (OSError, KeyError, ValueError):
                    continue
                vectors.append(sequence.mean(axis=0))
            if vectors:
                prototypes[label] = np.stack(vectors, axis=0).mean(axis=0)
        if {"you", "i"} - set(prototypes):
            return {}
        return prototypes

    def _load_like_i_location_prototypes(self) -> dict[str, np.ndarray]:
        if not self.enable_like_i_pairwise_calibration:
            return {}
        location_slice = self._location_feature_slice()
        if location_slice is None:
            return {}
        manifest_path = None
        processed_sequences_dir = None
        for dataset_root in self._prototype_dataset_roots():
            candidate_manifest = dataset_root / "manifests" / "sequence_manifest.csv"
            candidate_sequences = dataset_root / "processed_sequences"
            if candidate_manifest.exists() and candidate_sequences.exists():
                manifest_path = candidate_manifest
                processed_sequences_dir = candidate_sequences
                break
        if manifest_path is None or processed_sequences_dir is None:
            return {}
        try:
            manifest_rows = self._read_manifest_rows(manifest_path)
        except OSError:
            return {}
        chosen_paths: dict[str, list[Path]] = defaultdict(list)
        seen_groups: dict[str, set[str]] = defaultdict(set)
        for row in manifest_rows:
            label = str(row.get("english_label", "")).strip().lower()
            if label not in {"like", "i"}:
                continue
            if len(chosen_paths[label]) >= LIKE_I_LOCATION_PROTOTYPE_GROUPS:
                continue
            group_key = str(row.get("source_group_key", "")).strip()
            if group_key and group_key in seen_groups[label]:
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            sample_path = processed_sequences_dir / f"{sample_id}.npz"
            if not sample_path.exists():
                continue
            if group_key:
                seen_groups[label].add(group_key)
            chosen_paths[label].append(sample_path)
        prototypes: dict[str, np.ndarray] = {}
        for label, paths in chosen_paths.items():
            vectors: list[np.ndarray] = []
            for path in paths:
                try:
                    with np.load(path, allow_pickle=False) as payload:
                        sequence = payload["sequence"].astype(np.float32)[:, location_slice]
                except (OSError, KeyError, ValueError):
                    continue
                vectors.append(sequence.mean(axis=0))
            if vectors:
                prototypes[label] = np.stack(vectors, axis=0).mean(axis=0)
        if {"like", "i"} - set(prototypes):
            return {}
        return prototypes

    def _probabilities_from_logit_scores(self, logit_scores: dict[str, float]) -> dict[str, float]:
        ordered_labels = [self.index_to_label[int(index)] for index in range(len(self.index_to_label))]
        ordered_logits = np.asarray(
            [float(logit_scores.get(label, -1e9)) for label in ordered_labels],
            dtype=np.float64,
        )
        finite_mask = np.isfinite(ordered_logits)
        if not np.any(finite_mask):
            return {label: 0.0 for label in ordered_labels}
        stable_logits = ordered_logits.copy()
        stable_logits[finite_mask] -= np.max(stable_logits[finite_mask])
        exp_logits = np.zeros_like(stable_logits, dtype=np.float64)
        exp_logits[finite_mask] = np.exp(stable_logits[finite_mask])
        total = float(np.sum(exp_logits))
        if total <= 1e-12:
            return {label: 0.0 for label in ordered_labels}
        return {
            label: float(exp_logits[index] / total)
            for index, label in enumerate(ordered_labels)
        }

    def _maybe_apply_mother_nose_z_calibration(
        self,
        adjusted_sequence: np.ndarray,
        probabilities: dict[str, float],
        logit_scores: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not self.enable_mother_nose_z_calibration:
            return probabilities, logit_scores
        if self.pose_local_anchor != "torso_center":
            return probabilities, logit_scores
        if {"mother", "you"} - set(self.mother_you_left_face_prototypes):
            return probabilities, logit_scores
        top_candidates = self._top_candidates_from_probabilities(probabilities, limit=1)
        if not top_candidates or str(top_candidates[0][0]).strip().lower() != "you":
            return probabilities, logit_scores
        left_face_slice = self._left_face_feature_slice()
        if left_face_slice is None:
            return probabilities, logit_scores
        left_face_sequence = adjusted_sequence[:, left_face_slice].astype(np.float32, copy=False)
        if left_face_sequence.ndim != 2 or left_face_sequence.shape[1] != LEFT_FACE_VECTOR_DIM:
            return probabilities, logit_scores
        left_face_mean = left_face_sequence.mean(axis=0)
        mother_proto = self.mother_you_left_face_prototypes["mother"]
        you_proto = self.mother_you_left_face_prototypes["you"]
        baseline_gap = float(np.linalg.norm(left_face_mean - mother_proto) - np.linalg.norm(left_face_mean - you_proto))
        adjusted_mean = left_face_mean.copy()
        adjusted_mean[LEFT_FACE_NOSE_Z_INDEX] += float(self.mother_nose_z_offset)
        adjusted_gap = float(np.linalg.norm(adjusted_mean - mother_proto) - np.linalg.norm(adjusted_mean - you_proto))
        if not (baseline_gap > 0.0 and adjusted_gap < 0.0):
            return probabilities, logit_scores
        updated_logit_scores = dict(logit_scores)
        mother_logit = float(updated_logit_scores.get("mother", 0.0))
        you_logit = float(updated_logit_scores.get("you", 0.0))
        calibration_gain = max(baseline_gap - adjusted_gap, 0.0)
        updated_logit_scores["mother"] = max(mother_logit, you_logit) + calibration_gain
        updated_logit_scores["you"] = min(you_logit, mother_logit)
        updated_probabilities = self._probabilities_from_logit_scores(updated_logit_scores)
        return updated_probabilities, updated_logit_scores

    def _maybe_apply_you_i_leftbody_calibration(
        self,
        adjusted_sequence: np.ndarray,
        probabilities: dict[str, float],
        logit_scores: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not self.enable_you_i_leftbody_calibration:
            return probabilities, logit_scores
        if self.pose_local_anchor != "torso_center":
            return probabilities, logit_scores
        top_candidates = self._top_candidates_from_probabilities(probabilities, limit=1)
        if not top_candidates or str(top_candidates[0][0]).strip().lower() != "i":
            return probabilities, logit_scores
        gap = self._you_i_leftbody_gap(adjusted_sequence)
        if gap is None:
            return probabilities, logit_scores
        baseline_gap, adjusted_gap = gap
        if not (baseline_gap > 0.0 and adjusted_gap < 0.0):
            return probabilities, logit_scores
        updated_logit_scores = dict(logit_scores)
        you_logit = float(updated_logit_scores.get("you", 0.0))
        i_logit = float(updated_logit_scores.get("i", 0.0))
        calibration_gain = max(baseline_gap - adjusted_gap, 0.0)
        updated_logit_scores["you"] = max(you_logit, i_logit) + calibration_gain
        updated_logit_scores["i"] = min(you_logit, i_logit)
        updated_probabilities = self._probabilities_from_logit_scores(updated_logit_scores)
        return updated_probabilities, updated_logit_scores

    def _maybe_apply_you_like_pairwise_calibration(
        self,
        adjusted_sequence: np.ndarray,
        probabilities: dict[str, float],
        logit_scores: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        self._last_you_like_pairwise_debug = {
            "attempted": False,
            "applied": False,
            "delta": 0.0,
            "raw_confidence": 0.0,
            "top_margin": 0.0,
            "top_candidates": [],
        }
        if not self.enable_you_like_pairwise_calibration:
            return probabilities, logit_scores
        if self.mode != "trigger_based":
            return probabilities, logit_scores
        if self.pose_local_anchor != "torso_center":
            return probabilities, logit_scores
        top_candidates = self._top_candidates_from_probabilities(probabilities, limit=3)
        if len(top_candidates) < 2:
            return probabilities, logit_scores
        if str(top_candidates[0][0]).strip().lower() != "you":
            return probabilities, logit_scores
        if str(top_candidates[1][0]).strip().lower() != "like":
            return probabilities, logit_scores
        raw_confidence = float(top_candidates[0][1])
        top_margin = float(top_candidates[0][1] - top_candidates[1][1])
        if raw_confidence > self.trigger_emit_confidence_threshold and top_margin > self.trigger_min_top_margin:
            return probabilities, logit_scores
        gap = self._you_i_leftbody_gap(adjusted_sequence)
        if gap is None:
            return probabilities, logit_scores
        baseline_gap, adjusted_gap = gap
        if not (baseline_gap > 0.0 and adjusted_gap < 0.0):
            return probabilities, logit_scores
        self._last_you_like_pairwise_debug["attempted"] = True
        updated_logit_scores = dict(logit_scores)
        delta = max(float(self.you_like_pairwise_delta), 0.0)
        updated_logit_scores["you"] = float(updated_logit_scores.get("you", 0.0)) + (delta / 2.0)
        updated_logit_scores["like"] = float(updated_logit_scores.get("like", 0.0)) - (delta / 2.0)
        updated_probabilities = self._probabilities_from_logit_scores(updated_logit_scores)
        updated_top_candidates = self._top_candidates_from_probabilities(updated_probabilities, limit=3)
        updated_confidence = float(updated_top_candidates[0][1]) if updated_top_candidates else 0.0
        updated_margin = (
            float(updated_top_candidates[0][1] - updated_top_candidates[1][1])
            if len(updated_top_candidates) >= 2
            else updated_confidence
        )
        self._last_you_like_pairwise_debug.update(
            {
                "applied": True,
                "delta": float(delta),
                "raw_confidence": float(updated_confidence),
                "top_margin": float(updated_margin),
                "top_candidates": self._serialize_ranked_scores(updated_top_candidates, key_name="confidence", limit=3),
            }
        )
        return updated_probabilities, updated_logit_scores

    def _maybe_apply_like_i_pairwise_calibration(
        self,
        adjusted_sequence: np.ndarray,
        probabilities: dict[str, float],
        logit_scores: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        self._last_like_i_pairwise_debug = {
            "attempted": False,
            "applied": False,
            "delta": 0.0,
            "raw_confidence": 0.0,
            "top_margin": 0.0,
            "top_candidates": [],
        }
        if not self.enable_like_i_pairwise_calibration:
            return probabilities, logit_scores
        if self.mode != "trigger_based":
            return probabilities, logit_scores
        if self.pose_local_anchor != "torso_center":
            return probabilities, logit_scores
        top_candidates = self._top_candidates_from_probabilities(probabilities, limit=3)
        if len(top_candidates) < 2:
            return probabilities, logit_scores
        if str(top_candidates[0][0]).strip().lower() != "i":
            return probabilities, logit_scores
        if str(top_candidates[1][0]).strip().lower() != "like":
            return probabilities, logit_scores
        location_gap = self._like_i_location_gap(adjusted_sequence)
        if location_gap is None or location_gap >= 0.0:
            return probabilities, logit_scores
        self._last_like_i_pairwise_debug["attempted"] = True
        updated_logit_scores = dict(logit_scores)
        delta = max(float(self.like_i_pairwise_delta), 0.0)
        updated_logit_scores["like"] = float(updated_logit_scores.get("like", 0.0)) + (delta / 2.0)
        updated_logit_scores["i"] = float(updated_logit_scores.get("i", 0.0)) - (delta / 2.0)
        updated_probabilities = self._probabilities_from_logit_scores(updated_logit_scores)
        updated_top_candidates = self._top_candidates_from_probabilities(updated_probabilities, limit=3)
        updated_confidence = float(updated_top_candidates[0][1]) if updated_top_candidates else 0.0
        updated_margin = (
            float(updated_top_candidates[0][1] - updated_top_candidates[1][1])
            if len(updated_top_candidates) >= 2
            else updated_confidence
        )
        self._last_like_i_pairwise_debug.update(
            {
                "applied": True,
                "delta": float(delta),
                "raw_confidence": float(updated_confidence),
                "top_margin": float(updated_margin),
                "top_candidates": self._serialize_ranked_scores(updated_top_candidates, key_name="confidence", limit=3),
            }
        )
        return updated_probabilities, updated_logit_scores

    def _predict_sequence_scores(
        self,
        sequence: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float], list[tuple[str, float]], list[tuple[str, float]], str, float]:
        adjusted_sequence = apply_pose_hip_coordinate_scale(
            sequence,
            feature_spec=self.feature_spec,
            scale=self.pose_hip_coordinate_scale,
        )
        adjusted_sequence = apply_hand_mask_validity_scale(
            adjusted_sequence,
            feature_spec=self.feature_spec,
            scale=self.hand_mask_validity_scale,
        )
        tensor = torch.from_numpy(adjusted_sequence).unsqueeze(0).to(self.device)
        parts = split_feature_tensor(
            tensor,
            CONFIG.feature_mode,
            feature_spec=self.feature_spec,
        )
        with torch.no_grad():
            logits_tensor = self.model(
                parts["skeleton_stream"],
                parts["location_stream"],
                parts["motion_stream"],
            )
        probs_tensor = torch.softmax(logits_tensor, dim=-1)
        probs = probs_tensor.squeeze(0).detach().cpu().numpy()
        logits = logits_tensor.squeeze(0).detach().cpu().numpy()
        probabilities = {
            self.index_to_label[int(index)]: float(probs[int(index)])
            for index in range(len(probs))
        }
        logit_scores = {
            self.index_to_label[int(index)]: float(logits[int(index)])
            for index in range(len(logits))
        }
        probabilities, logit_scores = self._maybe_apply_mother_nose_z_calibration(
            adjusted_sequence,
            probabilities,
            logit_scores,
        )
        probabilities, logit_scores = self._maybe_apply_you_i_leftbody_calibration(
            adjusted_sequence,
            probabilities,
            logit_scores,
        )
        probabilities, logit_scores = self._maybe_apply_you_like_pairwise_calibration(
            adjusted_sequence,
            probabilities,
            logit_scores,
        )
        probabilities, logit_scores = self._maybe_apply_like_i_pairwise_calibration(
            adjusted_sequence,
            probabilities,
            logit_scores,
        )
        top_candidates = self._top_candidates_from_probabilities(probabilities)
        top_logits = self._top_logits_from_scores(logit_scores)
        raw_label, raw_confidence = top_candidates[0] if top_candidates else (CONFIG.no_sign_label, 0.0)
        return probabilities, logit_scores, top_candidates, top_logits, raw_label, float(raw_confidence)

    def _score_trigger_sequence_bundle(
        self,
        sequence: np.ndarray,
        *,
        sampled_path_sequence: np.ndarray | None,
        allow_want_anchor_fallback: bool = True,
    ) -> dict[str, object]:
        (
            probabilities,
            logit_scores,
            top_candidates,
            top_logits,
            raw_label,
            raw_confidence,
        ) = self._predict_sequence_scores(sequence)
        main_you_like_pairwise_debug = dict(self._last_you_like_pairwise_debug)
        main_like_i_pairwise_debug = dict(self._last_like_i_pairwise_debug)
        probabilities = self._apply_dynamic_nosign_suppression(probabilities)
        top_candidates = self._top_candidates_from_probabilities(probabilities)
        raw_label, raw_confidence = top_candidates[0] if top_candidates else (CONFIG.no_sign_label, 0.0)
        if len(top_candidates) >= 2:
            top_margin = float(top_candidates[0][1] - top_candidates[1][1])
        elif len(top_candidates) == 1:
            top_margin = float(top_candidates[0][1])
        else:
            top_margin = 0.0

        sampled_path_probabilities: dict[str, float] = {}
        sampled_path_top_candidates: list[tuple[str, float]] = []
        sampled_path_top_logits: list[tuple[str, float]] = []
        sampled_path_raw_label = CONFIG.no_sign_label
        sampled_path_raw_confidence = 0.0
        if isinstance(sampled_path_sequence, np.ndarray):
            (
                sampled_path_probabilities,
                _sampled_path_logit_scores,
                sampled_path_top_candidates,
                sampled_path_top_logits,
                sampled_path_raw_label,
                sampled_path_raw_confidence,
            ) = self._predict_sequence_scores(sampled_path_sequence)
            sampled_path_probabilities = self._apply_dynamic_nosign_suppression(sampled_path_probabilities)
            sampled_path_top_candidates = self._top_candidates_from_probabilities(sampled_path_probabilities)
            if sampled_path_top_candidates:
                sampled_path_raw_label, sampled_path_raw_confidence = sampled_path_top_candidates[0]

        return {
            "probabilities": probabilities,
            "logit_scores": logit_scores,
            "top_candidates": top_candidates,
            "top_logits": top_logits,
            "raw_label": raw_label,
            "raw_confidence": float(raw_confidence),
            "top_margin": float(top_margin),
            "sampled_path_probabilities": sampled_path_probabilities,
            "sampled_path_top_candidates": sampled_path_top_candidates,
            "sampled_path_top_logits": sampled_path_top_logits,
            "sampled_path_raw_label": sampled_path_raw_label,
            "sampled_path_raw_confidence": float(sampled_path_raw_confidence),
            "you_like_pairwise_debug": main_you_like_pairwise_debug,
            "like_i_pairwise_debug": main_like_i_pairwise_debug,
        }

    def _father_rule_hit_ratio(
        self,
        *,
        frames: list[TriggerFrameRecord],
        sampled_indices: list[int],
    ) -> tuple[float, int]:
        if not frames or not sampled_indices:
            return 0.0, 0
        hit_count = 0
        valid_count = 0
        prev_feature = None
        for sampled_index in sampled_indices:
            if sampled_index < 0 or sampled_index >= len(frames):
                continue
            frame_landmarks = frames[sampled_index].frame_landmarks
            built = build_frame_feature(
                left_hand=frame_landmarks.left_hand,
                right_hand=frame_landmarks.right_hand,
                pose=frame_landmarks.pose,
                mouth_center=frame_landmarks.mouth_center,
                chin=frame_landmarks.chin,
                left_hand_mask=frame_landmarks.left_hand_mask,
                right_hand_mask=frame_landmarks.right_hand_mask,
                pose_mask=frame_landmarks.pose_mask,
                mouth_mask=frame_landmarks.mouth_mask,
                chin_mask=frame_landmarks.chin_mask,
                prev_feature=prev_feature,
                feature_mode=CONFIG.feature_mode,
                feature_spec=self.feature_spec,
                pose_local_anchor=self.pose_local_anchor,
            )
            prev_feature = built
            right_valid = bool(built["right_hand_valid"][0] > 0)
            left_valid = bool(built["left_hand_valid"][0] > 0)
            nose_valid = bool(built["pose_mask"][0] > 0)
            if not nose_valid or (not right_valid and not left_valid):
                continue
            hand = built["right_hand"] if right_valid else built["left_hand"]
            hand_mask = built["right_hand_mask"] if right_valid else built["left_hand_mask"]
            if float(hand_mask[0]) <= 0.0:
                continue
            refs = built["reference_points"]
            wrist_y = float(hand[0][1])
            nose_y = float(refs["nose"][1])
            chin_valid = bool(built["chin_mask"][0] > 0)
            chin_y = float(refs["chin"][1]) if chin_valid else nose_y + 0.18
            face_span = max(chin_y - nose_y, 0.12)
            father_threshold = face_span * 0.60
            valid_count += 1
            if wrist_y - nose_y <= father_threshold:
                hit_count += 1
        if valid_count <= 0:
            return 0.0, 0
        return float(hit_count / valid_count), int(valid_count)

    def _maybe_apply_father_trigger_rescue(
        self,
        *,
        frames: list[TriggerFrameRecord],
        build_debug: dict[str, object],
        score_bundle: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        metadata = {
            "attempted": False,
            "applied": False,
            "hit_ratio": 0.0,
            "valid_count": 0,
            "pairwise_delta": 0.0,
            "raw_label": "",
            "raw_confidence": 0.0,
            "top_margin": 0.0,
            "top_candidates": [],
        }
        self._last_father_trigger_rescue_debug = dict(metadata)
        if not self.enable_father_trigger_rescue:
            return score_bundle, metadata
        if self.mode != "trigger_based":
            return score_bundle, metadata
        if self.pose_local_anchor != "torso_center":
            return score_bundle, metadata
        top_candidates = list(score_bundle.get("top_candidates", []))
        if len(top_candidates) < 2:
            return score_bundle, metadata
        raw_label = str(score_bundle.get("raw_label", "")).strip().lower()
        if raw_label != CONFIG.no_sign_label:
            return score_bundle, metadata
        if str(top_candidates[1][0]).strip().lower() != "father":
            return score_bundle, metadata
        raw_confidence = float(score_bundle.get("raw_confidence", 0.0))
        if raw_confidence > self.father_trigger_rescue_max_nosign_confidence:
            return score_bundle, metadata
        sampled_indices = [int(value) for value in build_debug.get("sampled_indices", [])]
        hit_ratio, valid_count = self._father_rule_hit_ratio(
            frames=frames,
            sampled_indices=sampled_indices,
        )
        metadata["attempted"] = True
        metadata["hit_ratio"] = float(hit_ratio)
        metadata["valid_count"] = int(valid_count)
        if valid_count <= 0 or hit_ratio < self.father_trigger_rescue_min_ratio:
            self._last_father_trigger_rescue_debug = dict(metadata)
            return score_bundle, metadata
        updated_logit_scores = dict(score_bundle.get("logit_scores", {}))
        delta = max(float(self.father_trigger_rescue_pairwise_delta), 0.0)
        updated_logit_scores["father"] = float(updated_logit_scores.get("father", 0.0)) + (delta / 2.0)
        updated_logit_scores[CONFIG.no_sign_label] = float(updated_logit_scores.get(CONFIG.no_sign_label, 0.0)) - (delta / 2.0)
        updated_probabilities = self._probabilities_from_logit_scores(updated_logit_scores)
        updated_top_candidates = self._top_candidates_from_probabilities(updated_probabilities)
        updated_top_logits = self._top_logits_from_scores(updated_logit_scores)
        updated_raw_label, updated_raw_confidence = (
            updated_top_candidates[0] if updated_top_candidates else (CONFIG.no_sign_label, 0.0)
        )
        if len(updated_top_candidates) >= 2:
            updated_top_margin = float(updated_top_candidates[0][1] - updated_top_candidates[1][1])
        elif len(updated_top_candidates) == 1:
            updated_top_margin = float(updated_top_candidates[0][1])
        else:
            updated_top_margin = 0.0
        metadata.update(
            {
                "pairwise_delta": float(delta),
                "raw_label": str(updated_raw_label),
                "raw_confidence": float(updated_raw_confidence),
                "top_margin": float(updated_top_margin),
                "top_candidates": self._serialize_ranked_scores(updated_top_candidates, key_name="confidence", limit=3),
            }
        )
        if str(updated_raw_label).strip().lower() != "father":
            self._last_father_trigger_rescue_debug = dict(metadata)
            return score_bundle, metadata
        if float(updated_raw_confidence) <= self.trigger_emit_confidence_threshold:
            self._last_father_trigger_rescue_debug = dict(metadata)
            return score_bundle, metadata
        if float(updated_top_margin) <= self.trigger_min_top_margin:
            self._last_father_trigger_rescue_debug = dict(metadata)
            return score_bundle, metadata
        metadata["applied"] = True
        self._last_father_trigger_rescue_debug = dict(metadata)
        updated_bundle = dict(score_bundle)
        updated_bundle.update(
            {
                "probabilities": updated_probabilities,
                "logit_scores": updated_logit_scores,
                "top_candidates": updated_top_candidates,
                "top_logits": updated_top_logits,
                "raw_label": str(updated_raw_label),
                "raw_confidence": float(updated_raw_confidence),
                "top_margin": float(updated_top_margin),
            }
        )
        return updated_bundle, metadata

    def _maybe_apply_want_anchor_fallback(
        self,
        *,
        frames: list[TriggerFrameRecord],
        build_debug: dict[str, object],
        score_bundle: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        metadata = {
            "attempted": False,
            "applied": False,
            "raw_label": "",
            "raw_confidence": 0.0,
            "top_margin": 0.0,
            "top_candidates": [],
            "candidate_anchor": "mid_shoulder",
        }
        self._last_want_anchor_fallback_debug = dict(metadata)
        if not self.enable_want_anchor_fallback:
            return score_bundle, build_debug, metadata
        if self.mode != "trigger_based":
            return score_bundle, build_debug, metadata
        if self.pose_local_anchor != "torso_center":
            return score_bundle, build_debug, metadata
        if not bool(build_debug.get("segment_has_both_hands", False)):
            return score_bundle, build_debug, metadata
        top_candidates = list(score_bundle.get("top_candidates", []))
        if len(top_candidates) < 3:
            return score_bundle, build_debug, metadata
        raw_label = str(score_bundle.get("raw_label", "")).strip().lower()
        raw_confidence = float(score_bundle.get("raw_confidence", 0.0))
        top_labels = [str(label).strip().lower() for label, _score in top_candidates[:3]]
        if raw_label != CONFIG.no_sign_label:
            return score_bundle, build_debug, metadata
        if top_labels[:2] != [CONFIG.no_sign_label, "student"]:
            return score_bundle, build_debug, metadata
        if "want" not in top_labels[:3]:
            return score_bundle, build_debug, metadata
        if raw_confidence > float(self.want_anchor_fallback_max_nosign_confidence):
            return score_bundle, build_debug, metadata
        metadata["attempted"] = True
        candidate_sequence, candidate_debug = self._build_trigger_sequence(
            frames,
            pose_local_anchor="mid_shoulder",
        )
        candidate_bundle = self._score_trigger_sequence_bundle(
            candidate_sequence,
            sampled_path_sequence=candidate_debug.get("sampled_path_sequence"),
            allow_want_anchor_fallback=False,
        )
        metadata["raw_label"] = str(candidate_bundle.get("raw_label", ""))
        metadata["raw_confidence"] = float(candidate_bundle.get("raw_confidence", 0.0))
        metadata["top_margin"] = float(candidate_bundle.get("top_margin", 0.0))
        metadata["top_candidates"] = self._serialize_ranked_scores(
            list(candidate_bundle.get("top_candidates", [])),
            key_name="confidence",
            limit=3,
        )
        candidate_label = str(candidate_bundle.get("raw_label", "")).strip().lower()
        if candidate_label == raw_label:
            self._last_want_anchor_fallback_debug = dict(metadata)
            return score_bundle, build_debug, metadata
        if candidate_label not in {"student", "want"}:
            self._last_want_anchor_fallback_debug = dict(metadata)
            return score_bundle, build_debug, metadata
        if float(candidate_bundle.get("raw_confidence", 0.0)) <= raw_confidence:
            self._last_want_anchor_fallback_debug = dict(metadata)
            return score_bundle, build_debug, metadata
        metadata["applied"] = True
        self._last_want_anchor_fallback_debug = dict(metadata)
        updated_debug = dict(build_debug)
        updated_debug.update(candidate_debug)
        return candidate_bundle, updated_debug, metadata

    def _maybe_apply_you_like_frontloaded_rescore(
        self,
        *,
        frames: list[TriggerFrameRecord],
        base_sequence: np.ndarray,
        build_debug: dict[str, object],
        score_bundle: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        metadata = {
            "attempted": False,
            "applied": False,
            "power": float(self.you_like_frontload_power),
            "raw_confidence": 0.0,
            "top_margin": 0.0,
            "top_candidates": [],
            "sampled_indices": [],
            "sampled_frame_indices": [],
        }
        if not self.enable_you_like_frontloaded_rescore:
            return score_bundle, build_debug, metadata
        if self.pose_local_anchor != "torso_center":
            return score_bundle, build_debug, metadata
        top_candidates = list(score_bundle.get("top_candidates", []))
        raw_label = str(score_bundle.get("raw_label", "")).strip().lower()
        raw_confidence = float(score_bundle.get("raw_confidence", 0.0))
        top_margin = float(score_bundle.get("top_margin", 0.0))
        if raw_label != "you" or len(top_candidates) < 2 or str(top_candidates[1][0]).strip().lower() != "like":
            return score_bundle, build_debug, metadata
        if raw_confidence > self.trigger_emit_confidence_threshold and top_margin > self.trigger_min_top_margin:
            return score_bundle, build_debug, metadata
        adjusted_sequence = apply_pose_hip_coordinate_scale(
            base_sequence,
            feature_spec=self.feature_spec,
            scale=self.pose_hip_coordinate_scale,
        )
        adjusted_sequence = apply_hand_mask_validity_scale(
            adjusted_sequence,
            feature_spec=self.feature_spec,
            scale=self.hand_mask_validity_scale,
        )
        gap = self._you_i_leftbody_gap(adjusted_sequence)
        if gap is None:
            return score_bundle, build_debug, metadata
        baseline_gap, adjusted_gap = gap
        if not (baseline_gap > 0.0 and adjusted_gap < 0.0):
            return score_bundle, build_debug, metadata
        feature_rows = build_debug.get("feature_rows")
        if not isinstance(feature_rows, list) or not feature_rows:
            return score_bundle, build_debug, metadata
        metadata["attempted"] = True
        sampled_index_list = self._frontloaded_trigger_sample_indices(
            len(frames),
            power=self.you_like_frontload_power,
        )
        if not sampled_index_list:
            return score_bundle, build_debug, metadata
        candidate_sequence, candidate_debug = self._materialize_trigger_sequence(
            frames=frames,
            feature_rows=feature_rows,
            sampled_index_list=sampled_index_list,
        )
        candidate_bundle = self._score_trigger_sequence_bundle(
            candidate_sequence,
            sampled_path_sequence=candidate_debug["sampled_path_sequence"],
        )
        metadata["raw_confidence"] = float(candidate_bundle["raw_confidence"])
        metadata["top_margin"] = float(candidate_bundle["top_margin"])
        metadata["top_candidates"] = self._serialize_ranked_scores(
            list(candidate_bundle["top_candidates"]),
            key_name="confidence",
        )
        metadata["sampled_indices"] = list(sampled_index_list)
        metadata["sampled_frame_indices"] = list(candidate_debug["sampled_frame_indices"])
        candidate_top_candidates = list(candidate_bundle.get("top_candidates", []))
        candidate_label = str(candidate_bundle.get("raw_label", "")).strip().lower()
        if candidate_label != "you":
            return score_bundle, build_debug, metadata
        if float(candidate_bundle["raw_confidence"]) <= raw_confidence:
            return score_bundle, build_debug, metadata
        if float(candidate_bundle["top_margin"]) <= top_margin:
            return score_bundle, build_debug, metadata
        if float(candidate_bundle["raw_confidence"]) <= self.trigger_emit_confidence_threshold:
            return score_bundle, build_debug, metadata
        if float(candidate_bundle["top_margin"]) <= self.trigger_min_top_margin:
            return score_bundle, build_debug, metadata
        metadata["applied"] = True
        updated_debug = dict(build_debug)
        updated_debug.update(candidate_debug)
        return candidate_bundle, updated_debug, metadata

    def _predict_probabilities(self, sequence: np.ndarray) -> tuple[dict[str, float], list[tuple[str, float]], str, float]:
        probabilities, _logits, top_candidates, _top_logits, raw_label, raw_confidence = self._predict_sequence_scores(
            sequence
        )
        return probabilities, top_candidates, raw_label, float(raw_confidence)

    def _update_trigger_buffer(
        self,
        frame_landmarks: FrameLandmarks,
        *,
        frame_index: int,
        motion_energy: float,
        left_hand_present: bool,
        right_hand_present: bool,
        pose_present: bool,
    ) -> tuple[str, list[TriggerFrameRecord] | None, TriggerSegmentDebug | None]:
        frame_record = self._make_trigger_frame_record(
            frame_index=frame_index,
            frame_landmarks=frame_landmarks,
            motion_energy=motion_energy,
            left_hand_present=left_hand_present,
            right_hand_present=right_hand_present,
            pose_present=pose_present,
        )
        has_hand = left_hand_present or right_hand_present
        if not self.trigger_active and self.trigger_pre_context_frames > 0:
            self.trigger_pre_buffer.append(frame_record)

        trigger_hit = has_hand and motion_energy >= self.trigger_motion_threshold
        hard_idle_hit = not has_hand
        hold_like_frame = has_hand and motion_energy <= self.trigger_idle_threshold

        if not self.trigger_active:
            self.trigger_start_count = self.trigger_start_count + 1 if trigger_hit else 0
            self.trigger_idle_count = 0
            if self.trigger_start_count >= self.trigger_start_frames:
                self.trigger_active = True
                self.trigger_segment_id += 1
                self.trigger_start_count = 0
                self.trigger_idle_count = 0
                self.trigger_current_pre_context_count = len(self.trigger_pre_buffer)
                self.trigger_action_buffer = list(self.trigger_pre_buffer)
                return "action_start", None, None
            return "idle", None, None

        self.trigger_action_buffer.append(frame_record)
        self.trigger_idle_count = self.trigger_idle_count + 1 if hard_idle_hit else 0

        reached_max_len = len(self.trigger_action_buffer) >= self.trigger_max_buffer_frames
        if self.trigger_idle_count >= self.trigger_end_frames or reached_max_len:
            end_reason = "idle_no_hand" if self.trigger_idle_count >= self.trigger_end_frames else "max_buffer"
            tail_slice = self.trigger_end_frames if end_reason == "idle_no_hand" else 0
            finalized_frames = (
                self.trigger_action_buffer[:-tail_slice]
                if tail_slice > 0 and len(self.trigger_action_buffer) > tail_slice
                else []
                if tail_slice > 0
                else list(self.trigger_action_buffer)
            )
            if finalized_frames and hold_like_frame and end_reason == "idle_no_hand":
                finalized_frames = list(self.trigger_action_buffer)
                tail_slice = 0
                end_reason = "idle_hold_preserved"
            carryover_frames: list[TriggerFrameRecord] = []
            if reached_max_len and self.trigger_pre_context_frames > 0:
                carryover_frames = list(finalized_frames[-self.trigger_pre_context_frames :])
            segment_debug = TriggerSegmentDebug(
                segment_id=self.trigger_segment_id,
                raw_start_frame=self.trigger_action_buffer[0].frame_index if self.trigger_action_buffer else frame_index,
                raw_end_frame=self.trigger_action_buffer[-1].frame_index if self.trigger_action_buffer else frame_index,
                start_frame=finalized_frames[0].frame_index if finalized_frames else frame_index,
                end_frame=finalized_frames[-1].frame_index if finalized_frames else frame_index,
                decision_frame=frame_index,
                raw_length=len(self.trigger_action_buffer),
                trimmed_length=len(finalized_frames),
                pre_context_frames=self.trigger_current_pre_context_count,
                tail_trimmed=max(0, len(self.trigger_action_buffer) - len(finalized_frames)),
                tail_trimmed_no_hand=max(0, tail_slice),
                end_reason=end_reason,
                trigger_end_reason=end_reason,
            )
            if len(finalized_frames) < self.trigger_min_action_frames:
                self.trigger_active = False
                self.trigger_idle_count = 0
                self.trigger_start_count = max(self.trigger_start_frames - 1, 0) if reached_max_len and carryover_frames else 0
                self.trigger_current_pre_context_count = 0
                self.trigger_pre_buffer = deque(carryover_frames, maxlen=self.trigger_pre_context_frames)
                self.trigger_action_buffer = []
                return "action_discarded", None, segment_debug
            self.trigger_active = False
            self.trigger_idle_count = 0
            self.trigger_start_count = max(self.trigger_start_frames - 1, 0) if reached_max_len and carryover_frames else 0
            self.trigger_current_pre_context_count = 0
            self.trigger_pre_buffer = deque(carryover_frames, maxlen=self.trigger_pre_context_frames)
            self.trigger_action_buffer = []
            return "action_end", finalized_frames, segment_debug

        return "collecting", None, None

    def _apply_dynamic_nosign_suppression(self, probabilities: dict[str, float]) -> dict[str, float]:
        if not self.enable_dynamic_nosign_suppression:
            return probabilities
        if CONFIG.no_sign_label not in probabilities:
            return probabilities
        penalty_factor = max(float(self.nosign_penalty_factor), 0.0)
        if abs(penalty_factor - 1.0) <= 1e-8:
            return probabilities
        adjusted = dict(probabilities)
        adjusted[CONFIG.no_sign_label] = float(adjusted[CONFIG.no_sign_label]) * penalty_factor
        total = float(sum(max(float(score), 0.0) for score in adjusted.values()))
        if total <= 1e-8:
            return probabilities
        return {
            label: float(max(float(score), 0.0) / total)
            for label, score in adjusted.items()
        }

    def _load_gesture_profiles(self, gesture_profile_path: str | Path | None) -> dict[str, dict]:
        if gesture_profile_path is None:
            return {}
        path = Path(gesture_profile_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        profiles = payload.get("profiles", {})
        return {str(label): dict(profile) for label, profile in profiles.items()}

    def _dominant_hand_data(self, built: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, bool]:
        right_valid = bool(built["right_hand_valid"][0] > 0)
        left_valid = bool(built["left_hand_valid"][0] > 0)
        if right_valid:
            return built["right_hand"], built["right_hand_mask"], True
        return built["left_hand"], built["left_hand_mask"], left_valid

    def _finger_extended(self, hand: np.ndarray, hand_mask: np.ndarray, tip_idx: int, pip_idx: int) -> bool:
        if hand_mask[tip_idx] <= 0 or hand_mask[pip_idx] <= 0 or hand_mask[0] <= 0:
            return False
        tip_to_wrist = float(np.linalg.norm(hand[tip_idx] - hand[0]))
        pip_to_wrist = float(np.linalg.norm(hand[pip_idx] - hand[0]))
        return tip_to_wrist > pip_to_wrist + 0.03

    def _profile_continuous_score(self, value: float, stats: dict[str, float]) -> float:
        median = float(stats.get("p50", stats.get("mean", value)))
        low = float(stats.get("p10", median))
        high = float(stats.get("p90", median))
        width = max(high - low, 0.12)
        distance = abs(value - median)
        score = 1.0 - min(distance / (width * 1.4), 1.0)
        return float(max(0.2, score))

    def _profile_binary_score(self, value: bool, stats: dict[str, float]) -> float:
        rate = float(stats.get("mean", 0.5))
        if rate >= 0.7:
            return 1.0 if value else 0.45
        if rate <= 0.3:
            return 1.0 if not value else 0.55
        return 0.9

    def _build_static_assist(self, built: dict[str, np.ndarray], probabilities: dict[str, float]) -> tuple[dict[str, float], str]:
        if not probabilities:
            return probabilities, ""
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = ranked[0]
        top_margin = float(top_score - ranked[1][1]) if len(ranked) >= 2 else float(top_score)
        if top_label in IGNORED_LABELS:
            return probabilities, ""
        if top_score < 0.20 or top_score > 0.80:
            return probabilities, ""
        if top_margin > 0.18:
            return probabilities, ""
        candidate_labels = {label for label, _score in ranked[:3]}
        if not (candidate_labels & ASSIST_TARGET_LABELS):
            return probabilities, ""
        if len(candidate_labels & ASSIST_TARGET_LABELS) < 2:
            return probabilities, ""
        if not self.gesture_profiles:
            return probabilities, ""

        hand, hand_mask, hand_valid = self._dominant_hand_data(built)
        if not hand_valid:
            return probabilities, ""

        refs = built["reference_points"]
        hand_center = built["right_hand_center"] if bool(built["right_hand_valid"][0] > 0) else built["left_hand_center"]
        chin_valid = bool(built["chin_mask"][0] > 0)
        mouth_valid = bool(built["mouth_mask"][0] > 0)
        thumb_tip = hand[4] if hand_mask[4] > 0 else hand_center
        index_extended = self._finger_extended(hand, hand_mask, 8, 6)
        middle_extended = self._finger_extended(hand, hand_mask, 12, 10)
        ring_extended = self._finger_extended(hand, hand_mask, 16, 14)
        pinky_extended = self._finger_extended(hand, hand_mask, 20, 18)
        extended_count = sum(int(flag) for flag in (index_extended, middle_extended, ring_extended, pinky_extended))
        open_palm = extended_count >= 3
        index_only = index_extended and not pinky_extended and extended_count <= 2
        pinky_only = pinky_extended and not index_extended and extended_count <= 2
        chin_distance = float(np.linalg.norm(hand_center - refs["chin"])) if chin_valid else 999.0
        mouth_distance = float(np.linalg.norm(hand_center - refs["mouth_center"])) if mouth_valid else 999.0
        chest_distance = float(np.linalg.norm(hand_center - refs["chest_center"]))
        thumb_chin_distance = float(np.linalg.norm(thumb_tip - refs["chin"])) if chin_valid else 999.0

        adjusted = dict(probabilities)
        debug_parts: list[str] = []
        shape_weight = float(self.gesture_assist_settings.get("assist_shape_weight", 0.85))
        position_weight = float(self.gesture_assist_settings.get("assist_position_weight", 0.15))
        multiplier_base = float(self.gesture_assist_settings.get("assist_multiplier_base", 0.92))
        multiplier_scale = float(self.gesture_assist_settings.get("assist_multiplier_scale", 0.18))
        multiplier_min = float(self.gesture_assist_settings.get("assist_multiplier_min", 0.92))
        multiplier_max = float(self.gesture_assist_settings.get("assist_multiplier_max", 1.10))
        current_features = {
            "center_x": hand_center[0],
            "center_y": hand_center[1],
            "chin_distance": chin_distance,
            "mouth_distance": mouth_distance,
            "chest_distance": chest_distance,
            "thumb_chin_distance": thumb_chin_distance,
            "index_only": index_only,
            "pinky_only": pinky_only,
            "open_palm": open_palm,
            "extended_count": float(extended_count),
        }
        for label in candidate_labels & ASSIST_TARGET_LABELS:
            profile = self.gesture_profiles.get(label)
            if not profile:
                continue
            feats = profile.get("features", {})
            handshape_scores = []
            position_scores = []
            for name in ("index_only", "pinky_only", "open_palm"):
                if name in feats:
                    handshape_scores.append(self._profile_binary_score(bool(current_features[name]), feats[name]))
            if "extended_count" in feats:
                handshape_scores.append(self._profile_continuous_score(float(current_features["extended_count"]), feats["extended_count"]))
            for name in ("center_y", "chin_distance", "mouth_distance", "chest_distance", "thumb_chin_distance"):
                if name in feats:
                    position_scores.append(self._profile_continuous_score(float(current_features[name]), feats[name]))
            shape_score = float(np.mean(handshape_scores)) if handshape_scores else 1.0
            position_score = float(np.mean(position_scores)) if position_scores else 1.0
            combined = (shape_weight * shape_score) + (position_weight * position_score)
            multiplier = float(min(multiplier_max, max(multiplier_min, multiplier_base + (combined - 0.5) * multiplier_scale)))
            adjusted[label] *= multiplier
            debug_parts.append(f"{label}:{multiplier:.2f}")

        total = float(sum(adjusted.values()))
        if total > 1e-8:
            adjusted = {label: float(score / total) for label, score in adjusted.items()}
        notes = (
            f"index_only={int(index_only)} pinky_only={int(pinky_only)} "
            f"open_palm={int(open_palm)} ext={extended_count} "
            + " ".join(debug_parts)
        ).strip()
        return adjusted, notes

    def _extract_gesture_features(self, built: dict[str, np.ndarray]) -> dict[str, float]:
        hand, hand_mask, hand_valid = self._dominant_hand_data(built)
        if not hand_valid:
            return {}
        refs = built["reference_points"]
        hand_center = built["right_hand_center"] if bool(built["right_hand_valid"][0] > 0) else built["left_hand_center"]
        chin_valid = bool(built["chin_mask"][0] > 0)
        mouth_valid = bool(built["mouth_mask"][0] > 0)
        thumb_tip = hand[4] if hand_mask[4] > 0 else hand_center
        index_extended = self._finger_extended(hand, hand_mask, 8, 6)
        middle_extended = self._finger_extended(hand, hand_mask, 12, 10)
        ring_extended = self._finger_extended(hand, hand_mask, 16, 14)
        pinky_extended = self._finger_extended(hand, hand_mask, 20, 18)
        extended_count = sum(int(flag) for flag in (index_extended, middle_extended, ring_extended, pinky_extended))
        open_palm = extended_count >= 3
        index_only = index_extended and not pinky_extended and extended_count <= 2
        pinky_only = pinky_extended and not index_extended and extended_count <= 2
        chin_distance = float(np.linalg.norm(hand_center - refs["chin"])) if chin_valid else 999.0
        mouth_distance = float(np.linalg.norm(hand_center - refs["mouth_center"])) if mouth_valid else 999.0
        chest_distance = float(np.linalg.norm(hand_center - refs["chest_center"]))
        thumb_chin_distance = float(np.linalg.norm(thumb_tip - refs["chin"])) if chin_valid else 999.0
        return {
            "center_x": float(hand_center[0]),
            "center_y": float(hand_center[1]),
            "chin_distance": float(chin_distance),
            "mouth_distance": float(mouth_distance),
            "chest_distance": float(chest_distance),
            "thumb_chin_distance": float(thumb_chin_distance),
            "index_only": float(index_only),
            "pinky_only": float(pinky_only),
            "open_palm": float(open_palm),
            "extended_count": float(extended_count),
        }

    def _profile_probability_scores(self, gesture_features: dict[str, float]) -> dict[str, float]:
        if not gesture_features or not self.gesture_profiles:
            return {}
        shape_weight = float(self.gesture_assist_settings.get("profile_shape_weight", 0.85))
        position_weight = float(self.gesture_assist_settings.get("profile_position_weight", 0.15))
        scores: dict[str, float] = {}
        for label, profile in self.gesture_profiles.items():
            feats = profile.get("features", {})
            handshape_scores = []
            position_scores = []
            for name in ("index_only", "pinky_only", "open_palm"):
                if name in feats:
                    handshape_scores.append(self._profile_binary_score(bool(gesture_features.get(name, 0.0) >= 0.5), feats[name]))
            if "extended_count" in feats:
                handshape_scores.append(self._profile_continuous_score(float(gesture_features.get("extended_count", 0.0)), feats["extended_count"]))
            for name in ("center_y", "chin_distance", "mouth_distance", "chest_distance", "thumb_chin_distance"):
                if name in feats:
                    position_scores.append(self._profile_continuous_score(float(gesture_features.get(name, 0.0)), feats[name]))
            shape_score = float(np.mean(handshape_scores)) if handshape_scores else 0.5
            position_score = float(np.mean(position_scores)) if position_scores else 0.5
            combined = max(0.05, (shape_weight * shape_score) + (position_weight * position_score))
            scores[label] = combined
        total = float(sum(scores.values()))
        if total > 1e-8:
            scores = {label: float(score / total) for label, score in scores.items()}
        return scores

    def reset(self) -> None:
        self.builder = SequenceBuilder(self.sequence_length)
        self.frame_valid_builder = SequenceBuilder(self.sequence_length)
        self.prev_feature = None
        self.postprocessor = InferencePostprocessor()
        self.frame_index = 0
        self.target_frame_size = None
        self.trigger_pre_buffer = deque(maxlen=self.trigger_pre_context_frames)
        self.trigger_action_buffer = []
        self.trigger_start_count = 0
        self.trigger_idle_count = 0
        self.trigger_active = False
        self.trigger_segment_id = 0
        self.trigger_current_pre_context_count = 0

    def close(self) -> None:
        self.extractor.close()

    def process_frame(self, frame: np.ndarray) -> FramePrediction:
        prepared_frame = self._prepare_frame(frame)
        frame_landmarks, results = self.extractor.extract(prepared_frame)
        built = build_frame_feature(
            left_hand=frame_landmarks.left_hand,
            right_hand=frame_landmarks.right_hand,
            pose=frame_landmarks.pose,
            mouth_center=frame_landmarks.mouth_center,
            chin=frame_landmarks.chin,
            left_hand_mask=frame_landmarks.left_hand_mask,
            right_hand_mask=frame_landmarks.right_hand_mask,
            pose_mask=frame_landmarks.pose_mask,
            mouth_mask=frame_landmarks.mouth_mask,
            chin_mask=frame_landmarks.chin_mask,
            prev_feature=self.prev_feature,
            feature_mode=CONFIG.feature_mode,
            feature_spec=self.feature_spec,
            pose_local_anchor=self.pose_local_anchor,
        )
        self.prev_feature = built
        display = self.extractor.draw(prepared_frame, results)
        left_hand_present = bool(built["frame_valid_mask"][0] > 0)
        right_hand_present = bool(built["frame_valid_mask"][1] > 0)
        pose_present = bool(built["frame_valid_mask"][2] > 0)
        motion_energy = self._motion_energy(built)
        frame_valid_ratio = float(np.mean(built["frame_valid_mask"])) if len(built["frame_valid_mask"]) else 0.0
        signal_score = self._signal_score(
            valid_ratio=frame_valid_ratio,
            left_hand_present=left_hand_present,
            right_hand_present=right_hand_present,
            pose_present=pose_present,
            motion_energy=motion_energy,
        )

        top_candidates: list[tuple[str, float]] = []
        probabilities: dict[str, float] = {}
        assisted_probabilities: dict[str, float] = {}
        assisted_top_candidates: list[tuple[str, float]] = []
        assist_notes = ""
        disambiguation_notes = ""
        disambiguation_applied = False
        top_margin = 0.0
        gesture_features: dict[str, float] = {}
        profile_probabilities: dict[str, float] = {}
        profile_top_candidates: list[tuple[str, float]] = []
        emitted_label = ""
        trigger_segment_debug: dict[str, object] | None = None

        if self.mode == "trigger_based":
            trigger_status, completed_sequence, completed_segment_debug = self._update_trigger_buffer(
                frame_landmarks,
                frame_index=self.frame_index,
                motion_energy=motion_energy,
                left_hand_present=left_hand_present,
                right_hand_present=right_hand_present,
                pose_present=pose_present,
            )
            if completed_sequence is None:
                if self.trigger_active:
                    status = "trigger_collecting"
                elif trigger_status == "action_discarded":
                    status = "trigger_discard"
                else:
                    status = "trigger_idle"
                decision = InferenceDecision(
                    final_label="collecting" if self.trigger_active else CONFIG.no_sign_label,
                    final_confidence=0.0,
                    raw_label="collecting" if self.trigger_active else CONFIG.no_sign_label,
                    raw_confidence=0.0,
                    status=status,
                    frame_valid=left_hand_present or right_hand_present,
                    sequence_valid=self.trigger_active,
                    valid_ratio=frame_valid_ratio,
                    smoothed_label="collecting" if self.trigger_active else CONFIG.no_sign_label,
                )
            else:
                aligned_sequence, build_debug = self._build_trigger_sequence(completed_sequence)
                score_bundle = self._score_trigger_sequence_bundle(
                    aligned_sequence,
                    sampled_path_sequence=build_debug.get("sampled_path_sequence"),
                )
                score_bundle, build_debug, frontload_metadata = self._maybe_apply_you_like_frontloaded_rescore(
                    frames=completed_sequence,
                    base_sequence=aligned_sequence,
                    build_debug=build_debug,
                    score_bundle=score_bundle,
                )
                score_bundle, build_debug, want_anchor_fallback_metadata = self._maybe_apply_want_anchor_fallback(
                    frames=completed_sequence,
                    build_debug=build_debug,
                    score_bundle=score_bundle,
                )
                score_bundle, father_trigger_rescue_metadata = self._maybe_apply_father_trigger_rescue(
                    frames=completed_sequence,
                    build_debug=build_debug,
                    score_bundle=score_bundle,
                )
                probabilities = dict(score_bundle["probabilities"])
                logit_scores = dict(score_bundle["logit_scores"])
                top_candidates = list(score_bundle["top_candidates"])
                top_logits = list(score_bundle["top_logits"])
                raw_label = str(score_bundle["raw_label"])
                raw_confidence = float(score_bundle["raw_confidence"])
                top_margin = float(score_bundle["top_margin"])
                sampled_path_top_candidates = list(score_bundle["sampled_path_top_candidates"])
                sampled_path_top_logits = list(score_bundle["sampled_path_top_logits"])
                sampled_path_raw_label = str(score_bundle["sampled_path_raw_label"])
                sampled_path_raw_confidence = float(score_bundle["sampled_path_raw_confidence"])
                you_like_pairwise_debug = dict(score_bundle.get("you_like_pairwise_debug", {}))
                like_i_pairwise_debug = dict(score_bundle.get("like_i_pairwise_debug", {}))
                assisted_probabilities = dict(probabilities)
                assisted_top_candidates = list(top_candidates)
                passes_margin = top_margin > self.trigger_min_top_margin
                if raw_label != CONFIG.no_sign_label and not passes_margin:
                    print(
                        f"[Rejected by Margin] {raw_label}: {raw_confidence:.4f}, margin: {top_margin:.4f}"
                    )
                emitted_label = (
                    raw_label
                    if (
                        raw_label != CONFIG.no_sign_label
                        and raw_confidence > self.trigger_emit_confidence_threshold
                        and passes_margin
                    )
                    else ""
                )
                status = "trigger_emit" if emitted_label else "trigger_reject"
                if completed_segment_debug is not None:
                    completed_segment_debug.raw_label = raw_label
                    completed_segment_debug.raw_confidence = raw_confidence
                    completed_segment_debug.top_margin = top_margin
                    completed_segment_debug.emitted_label = emitted_label
                    completed_segment_debug.decision_status = status
                    completed_segment_debug.raw_frame_indices = list(build_debug["raw_frame_indices"])
                    completed_segment_debug.sampled_indices = list(build_debug["sampled_indices"])
                    completed_segment_debug.sampled_frame_indices = list(build_debug["sampled_frame_indices"])
                    completed_segment_debug.motion_energy_curve = list(build_debug["motion_energy_curve"])
                    completed_segment_debug.top_candidates = self._serialize_ranked_scores(
                        top_candidates,
                        key_name="confidence",
                    )
                    completed_segment_debug.top_logits = self._serialize_ranked_scores(
                        top_logits,
                        key_name="logit",
                    )
                    completed_segment_debug.sampled_path_top_candidates = self._serialize_ranked_scores(
                        sampled_path_top_candidates,
                        key_name="confidence",
                    )
                    completed_segment_debug.sampled_path_top_logits = self._serialize_ranked_scores(
                        sampled_path_top_logits,
                        key_name="logit",
                    )
                    completed_segment_debug.sampled_path_raw_label = sampled_path_raw_label
                    completed_segment_debug.sampled_path_raw_confidence = sampled_path_raw_confidence
                    completed_segment_debug.feature_path_l2_mean = float(build_debug["feature_path_l2_mean"])
                    completed_segment_debug.feature_path_l2_max = float(build_debug["feature_path_l2_max"])
                    completed_segment_debug.feature_path_cosine_mean = float(build_debug["feature_path_cosine_mean"])
                    completed_segment_debug.segment_has_left_hand = bool(build_debug["segment_has_left_hand"])
                    completed_segment_debug.segment_has_right_hand = bool(build_debug["segment_has_right_hand"])
                    completed_segment_debug.segment_has_both_hands = bool(build_debug["segment_has_both_hands"])
                    completed_segment_debug.segment_pose_ratio = float(build_debug["segment_pose_ratio"])
                    completed_segment_debug.you_like_frontload_attempted = bool(frontload_metadata["attempted"])
                    completed_segment_debug.you_like_frontload_applied = bool(frontload_metadata["applied"])
                    completed_segment_debug.you_like_frontload_power = float(frontload_metadata["power"])
                    completed_segment_debug.you_like_frontload_top_margin = float(frontload_metadata["top_margin"])
                    completed_segment_debug.you_like_frontload_raw_confidence = float(frontload_metadata["raw_confidence"])
                    completed_segment_debug.you_like_frontload_top_candidates = list(frontload_metadata["top_candidates"])
                    completed_segment_debug.you_like_frontload_sampled_indices = list(frontload_metadata["sampled_indices"])
                    completed_segment_debug.you_like_frontload_sampled_frame_indices = list(
                        frontload_metadata["sampled_frame_indices"]
                    )
                    completed_segment_debug.you_like_pairwise_attempted = bool(you_like_pairwise_debug.get("attempted", False))
                    completed_segment_debug.you_like_pairwise_applied = bool(you_like_pairwise_debug.get("applied", False))
                    completed_segment_debug.you_like_pairwise_delta = float(you_like_pairwise_debug.get("delta", 0.0))
                    completed_segment_debug.you_like_pairwise_top_margin = float(
                        you_like_pairwise_debug.get("top_margin", 0.0)
                    )
                    completed_segment_debug.you_like_pairwise_raw_confidence = float(
                        you_like_pairwise_debug.get("raw_confidence", 0.0)
                    )
                    completed_segment_debug.you_like_pairwise_top_candidates = list(
                        you_like_pairwise_debug.get("top_candidates", [])
                    )
                    completed_segment_debug.like_i_pairwise_attempted = bool(like_i_pairwise_debug.get("attempted", False))
                    completed_segment_debug.like_i_pairwise_applied = bool(like_i_pairwise_debug.get("applied", False))
                    completed_segment_debug.like_i_pairwise_delta = float(like_i_pairwise_debug.get("delta", 0.0))
                    completed_segment_debug.like_i_pairwise_top_margin = float(
                        like_i_pairwise_debug.get("top_margin", 0.0)
                    )
                    completed_segment_debug.like_i_pairwise_raw_confidence = float(
                        like_i_pairwise_debug.get("raw_confidence", 0.0)
                    )
                    completed_segment_debug.like_i_pairwise_top_candidates = list(
                        like_i_pairwise_debug.get("top_candidates", [])
                    )
                    completed_segment_debug.want_anchor_fallback_attempted = bool(
                        want_anchor_fallback_metadata.get("attempted", False)
                    )
                    completed_segment_debug.want_anchor_fallback_applied = bool(
                        want_anchor_fallback_metadata.get("applied", False)
                    )
                    completed_segment_debug.want_anchor_fallback_raw_label = str(
                        want_anchor_fallback_metadata.get("raw_label", "")
                    )
                    completed_segment_debug.want_anchor_fallback_raw_confidence = float(
                        want_anchor_fallback_metadata.get("raw_confidence", 0.0)
                    )
                    completed_segment_debug.want_anchor_fallback_top_margin = float(
                        want_anchor_fallback_metadata.get("top_margin", 0.0)
                    )
                    completed_segment_debug.want_anchor_fallback_top_candidates = list(
                        want_anchor_fallback_metadata.get("top_candidates", [])
                    )
                    completed_segment_debug.want_anchor_fallback_candidate_anchor = str(
                        want_anchor_fallback_metadata.get("candidate_anchor", "")
                    )
                    completed_segment_debug.father_trigger_rescue_attempted = bool(
                        father_trigger_rescue_metadata.get("attempted", False)
                    )
                    completed_segment_debug.father_trigger_rescue_applied = bool(
                        father_trigger_rescue_metadata.get("applied", False)
                    )
                    completed_segment_debug.father_trigger_rescue_hit_ratio = float(
                        father_trigger_rescue_metadata.get("hit_ratio", 0.0)
                    )
                    completed_segment_debug.father_trigger_rescue_valid_count = int(
                        father_trigger_rescue_metadata.get("valid_count", 0)
                    )
                    completed_segment_debug.father_trigger_rescue_pairwise_delta = float(
                        father_trigger_rescue_metadata.get("pairwise_delta", 0.0)
                    )
                    completed_segment_debug.father_trigger_rescue_raw_label = str(
                        father_trigger_rescue_metadata.get("raw_label", "")
                    )
                    completed_segment_debug.father_trigger_rescue_raw_confidence = float(
                        father_trigger_rescue_metadata.get("raw_confidence", 0.0)
                    )
                    completed_segment_debug.father_trigger_rescue_top_margin = float(
                        father_trigger_rescue_metadata.get("top_margin", 0.0)
                    )
                    completed_segment_debug.father_trigger_rescue_top_candidates = list(
                        father_trigger_rescue_metadata.get("top_candidates", [])
                    )
                    trigger_segment_debug = completed_segment_debug.to_dict()
                final_label = emitted_label or CONFIG.no_sign_label
                final_confidence = raw_confidence if emitted_label else 0.0
                decision = InferenceDecision(
                    final_label=final_label,
                    final_confidence=final_confidence,
                    raw_label=raw_label,
                    raw_confidence=raw_confidence,
                    status=status,
                    frame_valid=left_hand_present or right_hand_present,
                    sequence_valid=True,
                    valid_ratio=frame_valid_ratio,
                    smoothed_label=final_label,
                )
        else:
            self.builder.append(built["feature_vector"])
            self.frame_valid_builder.append(built["frame_valid_mask"])
            if self.builder.sliding_window() is None:
                decision = InferenceDecision(
                    final_label="collecting",
                    final_confidence=0.0,
                    raw_label="collecting",
                    raw_confidence=0.0,
                    status="collecting",
                    frame_valid=False,
                    sequence_valid=False,
                    valid_ratio=0.0,
                    smoothed_label="collecting",
                )
            else:
                sequence = self.builder.sliding_window()
                probabilities, top_candidates, raw_label, raw_confidence = self._predict_probabilities(sequence)
                probabilities = self._apply_dynamic_nosign_suppression(probabilities)
                top_candidates = self._top_candidates_from_probabilities(probabilities)
                raw_label, raw_confidence = top_candidates[0] if top_candidates else ("collecting", 0.0)
                if self.enable_rule_based_disambiguation:
                    disambiguation = self.disambiguator.apply(probabilities, built)
                    probabilities = dict(disambiguation.probabilities)
                    raw_label = disambiguation.final_label
                    raw_confidence = disambiguation.final_confidence
                    disambiguation_notes = disambiguation.notes
                    disambiguation_applied = disambiguation.applied
                else:
                    disambiguation_notes = "bypassed: rule-based disambiguation disabled"
                    disambiguation_applied = False
                top_candidates = self._top_candidates_from_probabilities(probabilities)
                assisted_probabilities, assist_notes = self._build_static_assist(built, probabilities)
                if assisted_probabilities:
                    assisted_top_candidates = self._top_candidates_from_probabilities(assisted_probabilities)
                else:
                    assisted_probabilities = dict(probabilities)
                    assisted_top_candidates = list(top_candidates)
                gesture_features = self._extract_gesture_features(built)
                profile_probabilities = self._profile_probability_scores(gesture_features)
                profile_top_candidates = sorted(profile_probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
                if len(assisted_top_candidates) >= 2:
                    top_margin = float(assisted_top_candidates[0][1] - assisted_top_candidates[1][1])
                elif len(assisted_top_candidates) == 1:
                    top_margin = float(assisted_top_candidates[0][1])
                thresholded_label, threshold_status = apply_confidence_threshold(
                    raw_label,
                    raw_confidence,
                    self.confidence_threshold,
                )
                decision = self.postprocessor.decide(
                    raw_label=raw_label,
                    raw_confidence=raw_confidence,
                    frame_valid_mask=built["frame_valid_mask"],
                    frame_valid_sequence=self.frame_valid_builder.sliding_window(),
                    thresholded_label=thresholded_label,
                    threshold_status=threshold_status,
                )

        blank_like_status = decision.status in {"invalid_frame", "insufficient_signal", "collecting"}
        if blank_like_status:
            line1 = f"Prediction: {CONFIG.no_sign_label} (0.00)"
            line2 = "Raw: idle"
            line3 = "Top3: -"
            line4 = "Assist: -"
        else:
            line1 = f"Prediction: {decision.final_label} ({decision.final_confidence:.2f})"
            line2 = f"Raw: {decision.raw_label} ({decision.raw_confidence:.2f})"
            top3_text = ", ".join(f"{label}:{score:.2f}" for label, score in top_candidates[:3]) if top_candidates else "-"
            line3 = f"Top3: {top3_text}"
            assist_text = ", ".join(f"{label}:{score:.2f}" for label, score in assisted_top_candidates[:3]) if assisted_top_candidates else "-"
            line4 = f"Assist: {assist_text}"
        line5 = f"Status: {decision.status} valid={decision.valid_ratio:.2f}"
        line6 = f"Mode={CONFIG.feature_mode} Branch={self.checkpoint_branch} Frame={self.frame_index}"
        cv2.putText(display, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
        cv2.putText(display, line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, line3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, line4, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, line5, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, line6, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        prediction = FramePrediction(
            display_frame=display,
            decision=decision,
            engine_mode=self.mode,
            emitted_label=emitted_label,
            checkpoint_branch=self.checkpoint_branch,
            frame_index=self.frame_index,
            top_candidates=top_candidates,
            probabilities=probabilities,
            assisted_top_candidates=assisted_top_candidates,
            assisted_probabilities=assisted_probabilities,
            assist_notes=assist_notes,
            disambiguation_notes=disambiguation_notes,
            disambiguation_applied=disambiguation_applied,
            left_hand_present=left_hand_present,
            right_hand_present=right_hand_present,
            pose_present=pose_present,
            motion_energy=motion_energy,
            top_margin=top_margin,
            signal_score=signal_score,
            gesture_features=gesture_features,
            profile_probabilities=profile_probabilities,
            profile_top_candidates=profile_top_candidates,
            trigger_segment_debug=trigger_segment_debug,
        )
        self.frame_index += 1
        return prediction

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.target_frame_size is None:
            self.target_frame_size = (int(frame.shape[1]), int(frame.shape[0]))
            return frame
        width, height = self.target_frame_size
        if frame.shape[1] == width and frame.shape[0] == height:
            return frame
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
