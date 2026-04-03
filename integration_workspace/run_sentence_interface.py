from __future__ import annotations

import copy
import csv
import importlib.util
import json
import queue
import threading
import time
import tkinter as tk
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from src.app.sentence_composer import SentenceComposer
from src.app.sign_sentence_engine import MultibranchSequenceEngine, PhraseCollector
from src.app.scripted_video_alignment import ScriptedFrameSignal, build_scripted_alignment, parse_script_tokens_from_path
from src.utils.paths import ensure_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_TUNING_SETTINGS_PATH = PROJECT_ROOT / "realtime_tuning_settings.py"


def _deep_update(base: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_user_tuning_settings() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], list[str]]:
    preset_overrides: dict[str, dict[str, object]] = {}
    decoder_overrides: dict[str, dict[str, object]] = {}
    notices: list[str] = []
    if not USER_TUNING_SETTINGS_PATH.exists():
        return preset_overrides, decoder_overrides, notices
    try:
        spec = importlib.util.spec_from_file_location("user_realtime_tuning_settings", USER_TUNING_SETTINGS_PATH)
        if spec is None or spec.loader is None:
            notices.append(f"無法讀取設定檔：{USER_TUNING_SETTINGS_PATH}")
            return preset_overrides, decoder_overrides, notices
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        preset_overrides = copy.deepcopy(getattr(module, "MODE_PRESET_OVERRIDES", {}))
        decoder_overrides = copy.deepcopy(getattr(module, "DECODER_PRESET_OVERRIDES", {}))
        notices.append(f"已載入根目錄設定檔：{USER_TUNING_SETTINGS_PATH.name}")
    except Exception as exc:
        notices.append(f"設定檔載入失敗，已改用內建預設：{exc}")
    return preset_overrides, decoder_overrides, notices


def get_decoder_settings_for_mode(mode_name: str, *, confidence_threshold: float | None = None) -> dict[str, object]:
    decoder_presets = copy.deepcopy(SentenceInterfaceApp.DEFAULT_DECODER_PRESETS)
    _preset_overrides, decoder_overrides, _notices = load_user_tuning_settings()
    decoder_presets = _deep_update(decoder_presets, decoder_overrides)
    settings = copy.deepcopy(decoder_presets.get("default", {}))
    mode_overrides = decoder_presets.get(mode_name, {})
    settings.update(mode_overrides)
    if "min_confidence" not in settings:
        fallback_confidence = max(float(confidence_threshold or 0.0), 0.45)
        settings["min_confidence"] = float(fallback_confidence)
    else:
        settings["min_confidence"] = float(settings["min_confidence"])
    return settings


class RealtimeTestRecorder:
    def __init__(
        self,
        *,
        output_root: Path,
        mode_name: str,
        checkpoint_path: str,
        confidence_threshold: float,
        sequence_length: int,
        mirror_input: bool,
        camera_index: int,
        gesture_profile_path: str = "",
        decoder_params: dict[str, object] | None = None,
        source_video_path: str = "",
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = ensure_dir(output_root / f"realtime_test_{timestamp}")
        self.video_path = self.session_dir / "preview_video.mp4"
        self.frame_csv_path = self.session_dir / "frame_predictions.csv"
        self.trigger_segments_csv_path = self.session_dir / "trigger_segments.csv"
        self.params_json_path = self.session_dir / "session_params.json"
        self.summary_json_path = self.session_dir / "session_summary.json"
        self._writer = None
        self._rows: list[dict[str, object]] = []
        self._trigger_segments: list[dict[str, object]] = []
        self._started_at = datetime.now().isoformat(timespec="seconds")

        self.params = {
            "mode_name": mode_name,
            "checkpoint_path": checkpoint_path,
            "confidence_threshold": confidence_threshold,
            "sequence_length": sequence_length,
            "mirror_input": mirror_input,
            "camera_index": camera_index,
            "gesture_profile_path": gesture_profile_path,
            "decoder_params": decoder_params or {},
            "source_video_path": source_video_path,
            "session_started_at": self._started_at,
        }
        self.params_json_path.write_text(json.dumps(self.params, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_frame(
        self,
        frame,
        *,
        frame_index: int,
        raw_label: str,
        raw_confidence: float,
        final_label: str,
        final_confidence: float,
        status: str,
        valid_ratio: float,
        top_candidates: list[tuple[str, float]],
        stream_label: str = "",
        stream_confidence: float = 0.0,
        decoder_state: str = "",
        emitted_label: str = "",
        assisted_top_candidates: list[tuple[str, float]] | None = None,
        assist_notes: str = "",
        probabilities: dict[str, float] | None = None,
        assisted_probabilities: dict[str, float] | None = None,
        disambiguation_notes: str = "",
        disambiguation_applied: bool = False,
        motion_energy: float = 0.0,
        top_margin: float = 0.0,
        signal_score: float = 0.0,
        left_hand_present: bool = False,
        right_hand_present: bool = False,
        pose_present: bool = False,
        trigger_segment_debug: dict[str, object] | None = None,
    ) -> None:
        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                (width, height),
            )
        if self._writer is not None and self._writer.isOpened():
            self._writer.write(frame)
        self._rows.append(
            {
                "frame_index": frame_index,
                "raw_label": raw_label,
                "raw_confidence": f"{raw_confidence:.6f}",
                "final_label": final_label,
                "final_confidence": f"{final_confidence:.6f}",
                "status": status,
                "valid_ratio": f"{valid_ratio:.6f}",
                "stream_label": stream_label,
                "stream_confidence": f"{stream_confidence:.6f}",
                "decoder_state": decoder_state,
                "emitted_label": emitted_label,
                "assisted_top_candidates": json.dumps(
                    [
                        {"label": label, "confidence": round(confidence, 6)}
                        for label, confidence in (assisted_top_candidates or [])
                    ],
                    ensure_ascii=False,
                ),
                "assist_notes": assist_notes,
                "probabilities": json.dumps(probabilities or {}, ensure_ascii=False),
                "assisted_probabilities": json.dumps(assisted_probabilities or {}, ensure_ascii=False),
                "disambiguation_notes": disambiguation_notes,
                "disambiguation_applied": int(bool(disambiguation_applied)),
                "motion_energy": f"{motion_energy:.6f}",
                "top_margin": f"{top_margin:.6f}",
                "signal_score": f"{signal_score:.6f}",
                "left_hand_present": int(bool(left_hand_present)),
                "right_hand_present": int(bool(right_hand_present)),
                "pose_present": int(bool(pose_present)),
                "top_candidates": json.dumps(
                    [{"label": label, "confidence": round(confidence, 6)} for label, confidence in top_candidates],
                    ensure_ascii=False,
                ),
            }
        )
        if trigger_segment_debug:
            self._trigger_segments.append(
                {
                    "segment_id": int(trigger_segment_debug.get("segment_id", 0) or 0),
                    "raw_start_frame": int(trigger_segment_debug.get("raw_start_frame", frame_index) or frame_index),
                    "raw_end_frame": int(trigger_segment_debug.get("raw_end_frame", frame_index) or frame_index),
                    "start_frame": int(trigger_segment_debug.get("start_frame", frame_index) or frame_index),
                    "end_frame": int(trigger_segment_debug.get("end_frame", frame_index) or frame_index),
                    "decision_frame": int(trigger_segment_debug.get("decision_frame", frame_index) or frame_index),
                    "raw_length": int(trigger_segment_debug.get("raw_length", 0) or 0),
                    "trimmed_length": int(trigger_segment_debug.get("trimmed_length", 0) or 0),
                    "pre_context_frames": int(trigger_segment_debug.get("pre_context_frames", 0) or 0),
                    "tail_trimmed": int(trigger_segment_debug.get("tail_trimmed", 0) or 0),
                    "tail_trimmed_no_hand": int(trigger_segment_debug.get("tail_trimmed_no_hand", 0) or 0),
                    "end_reason": str(trigger_segment_debug.get("end_reason", "")),
                    "trigger_end_reason": str(trigger_segment_debug.get("trigger_end_reason", "")),
                    "raw_label": str(trigger_segment_debug.get("raw_label", raw_label)),
                    "raw_confidence": f"{float(trigger_segment_debug.get('raw_confidence', raw_confidence) or 0.0):.6f}",
                    "top_margin": f"{float(trigger_segment_debug.get('top_margin', top_margin) or 0.0):.6f}",
                    "emitted_label": str(trigger_segment_debug.get("emitted_label", emitted_label)),
                    "decision_status": str(trigger_segment_debug.get("decision_status", status)),
                    "raw_frame_indices": json.dumps(trigger_segment_debug.get("raw_frame_indices", []), ensure_ascii=False),
                    "sampled_indices": json.dumps(trigger_segment_debug.get("sampled_indices", []), ensure_ascii=False),
                    "sampled_frame_indices": json.dumps(trigger_segment_debug.get("sampled_frame_indices", []), ensure_ascii=False),
                    "motion_energy_curve": json.dumps(trigger_segment_debug.get("motion_energy_curve", []), ensure_ascii=False),
                    "top_candidates": json.dumps(trigger_segment_debug.get("top_candidates", []), ensure_ascii=False),
                    "top_logits": json.dumps(trigger_segment_debug.get("top_logits", []), ensure_ascii=False),
                    "sampled_path_top_candidates": json.dumps(trigger_segment_debug.get("sampled_path_top_candidates", []), ensure_ascii=False),
                    "sampled_path_top_logits": json.dumps(trigger_segment_debug.get("sampled_path_top_logits", []), ensure_ascii=False),
                    "sampled_path_raw_label": str(trigger_segment_debug.get("sampled_path_raw_label", "")),
                    "sampled_path_raw_confidence": f"{float(trigger_segment_debug.get('sampled_path_raw_confidence', 0.0) or 0.0):.6f}",
                    "feature_path_l2_mean": f"{float(trigger_segment_debug.get('feature_path_l2_mean', 0.0) or 0.0):.6f}",
                    "feature_path_l2_max": f"{float(trigger_segment_debug.get('feature_path_l2_max', 0.0) or 0.0):.6f}",
                    "feature_path_cosine_mean": f"{float(trigger_segment_debug.get('feature_path_cosine_mean', 0.0) or 0.0):.6f}",
                    "segment_has_left_hand": int(bool(trigger_segment_debug.get("segment_has_left_hand", 0))),
                    "segment_has_right_hand": int(bool(trigger_segment_debug.get("segment_has_right_hand", 0))),
                    "segment_has_both_hands": int(bool(trigger_segment_debug.get("segment_has_both_hands", 0))),
                    "segment_pose_ratio": f"{float(trigger_segment_debug.get('segment_pose_ratio', 0.0) or 0.0):.6f}",
                    "father_trigger_rescue_attempted": int(bool(trigger_segment_debug.get("father_trigger_rescue_attempted", 0))),
                    "father_trigger_rescue_applied": int(bool(trigger_segment_debug.get("father_trigger_rescue_applied", 0))),
                    "father_trigger_rescue_hit_ratio": f"{float(trigger_segment_debug.get('father_trigger_rescue_hit_ratio', 0.0) or 0.0):.6f}",
                    "father_trigger_rescue_valid_count": int(trigger_segment_debug.get("father_trigger_rescue_valid_count", 0) or 0),
                    "father_trigger_rescue_pairwise_delta": f"{float(trigger_segment_debug.get('father_trigger_rescue_pairwise_delta', 0.0) or 0.0):.6f}",
                    "father_trigger_rescue_raw_label": str(trigger_segment_debug.get("father_trigger_rescue_raw_label", "")),
                    "father_trigger_rescue_raw_confidence": f"{float(trigger_segment_debug.get('father_trigger_rescue_raw_confidence', 0.0) or 0.0):.6f}",
                    "father_trigger_rescue_top_margin": f"{float(trigger_segment_debug.get('father_trigger_rescue_top_margin', 0.0) or 0.0):.6f}",
                    "father_trigger_rescue_top_candidates": json.dumps(trigger_segment_debug.get("father_trigger_rescue_top_candidates", []), ensure_ascii=False),
                }
            )

    def close(self) -> dict[str, str]:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        with self.frame_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "frame_index",
                    "raw_label",
                    "raw_confidence",
                    "final_label",
                    "final_confidence",
                    "status",
                    "valid_ratio",
                    "stream_label",
                    "stream_confidence",
                    "decoder_state",
                    "emitted_label",
                    "assisted_top_candidates",
                    "assist_notes",
                    "probabilities",
                    "assisted_probabilities",
                    "disambiguation_notes",
                    "disambiguation_applied",
                    "motion_energy",
                    "top_margin",
                    "signal_score",
                    "left_hand_present",
                    "right_hand_present",
                    "pose_present",
                    "top_candidates",
                ],
            )
            writer.writeheader()
            writer.writerows(self._rows)

        if self._trigger_segments:
            with self.trigger_segments_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "segment_id",
                        "raw_start_frame",
                        "raw_end_frame",
                        "start_frame",
                        "end_frame",
                        "decision_frame",
                        "raw_length",
                        "trimmed_length",
                        "pre_context_frames",
                        "tail_trimmed",
                        "tail_trimmed_no_hand",
                        "end_reason",
                        "trigger_end_reason",
                        "raw_label",
                        "raw_confidence",
                        "top_margin",
                        "emitted_label",
                        "decision_status",
                        "raw_frame_indices",
                        "sampled_indices",
                        "sampled_frame_indices",
                        "motion_energy_curve",
                        "top_candidates",
                        "top_logits",
                        "sampled_path_top_candidates",
                        "sampled_path_top_logits",
                        "sampled_path_raw_label",
                        "sampled_path_raw_confidence",
                        "feature_path_l2_mean",
                        "feature_path_l2_max",
                        "feature_path_cosine_mean",
                        "segment_has_left_hand",
                        "segment_has_right_hand",
                        "segment_has_both_hands",
                        "segment_pose_ratio",
                        "father_trigger_rescue_attempted",
                        "father_trigger_rescue_applied",
                        "father_trigger_rescue_hit_ratio",
                        "father_trigger_rescue_valid_count",
                        "father_trigger_rescue_pairwise_delta",
                        "father_trigger_rescue_raw_label",
                        "father_trigger_rescue_raw_confidence",
                        "father_trigger_rescue_top_margin",
                        "father_trigger_rescue_top_candidates",
                    ],
                )
                writer.writeheader()
                writer.writerows(self._trigger_segments)

        status_counts = Counter(str(row["status"]) for row in self._rows)
        raw_label_counts = Counter(str(row["raw_label"]) for row in self._rows)
        stream_label_counts = Counter(str(row["stream_label"]) for row in self._rows if str(row["stream_label"]))
        decoder_state_counts = Counter(str(row["decoder_state"]) for row in self._rows if str(row["decoder_state"]))
        emitted_label_counts = Counter(str(row["emitted_label"]) for row in self._rows if str(row["emitted_label"]))
        valid_ratios = [float(row["valid_ratio"]) for row in self._rows] if self._rows else []
        motion_energies = [float(row["motion_energy"]) for row in self._rows] if self._rows else []
        top_margins = [float(row["top_margin"]) for row in self._rows] if self._rows else []
        signal_scores = [float(row["signal_score"]) for row in self._rows] if self._rows else []
        trigger_end_reason_counts = (
            Counter(str(row["end_reason"]) for row in self._trigger_segments if str(row.get("end_reason", "")))
            if self._trigger_segments
            else Counter()
        )
        left_hand_ratio = (
            sum(int(row["left_hand_present"]) for row in self._rows) / max(len(self._rows), 1)
            if self._rows
            else 0.0
        )
        right_hand_ratio = (
            sum(int(row["right_hand_present"]) for row in self._rows) / max(len(self._rows), 1)
            if self._rows
            else 0.0
        )
        pose_ratio = (
            sum(int(row["pose_present"]) for row in self._rows) / max(len(self._rows), 1)
            if self._rows
            else 0.0
        )
        invalid_streak = 0
        max_invalid_streak = 0
        for row in self._rows:
            if str(row["status"]) in {"invalid_frame", "insufficient_signal"}:
                invalid_streak += 1
                max_invalid_streak = max(max_invalid_streak, invalid_streak)
            else:
                invalid_streak = 0
        duration_minutes = len(self._rows) / 20.0 / 60.0 if self._rows else 0.0
        summary = {
            **self.params,
            "session_finished_at": datetime.now().isoformat(timespec="seconds"),
            "frame_count": len(self._rows),
            "video_path": str(self.video_path),
            "frame_csv_path": str(self.frame_csv_path),
            "trigger_segments_csv_path": str(self.trigger_segments_csv_path) if self._trigger_segments else "",
            "status_counts": dict(status_counts),
            "raw_label_counts": dict(raw_label_counts),
            "stream_label_counts": dict(stream_label_counts),
            "decoder_state_counts": dict(decoder_state_counts),
            "emitted_label_counts": dict(emitted_label_counts),
            "avg_valid_ratio": (sum(valid_ratios) / len(valid_ratios)) if valid_ratios else 0.0,
            "avg_motion_energy": (sum(motion_energies) / len(motion_energies)) if motion_energies else 0.0,
            "avg_top_margin": (sum(top_margins) / len(top_margins)) if top_margins else 0.0,
            "avg_signal_score": (sum(signal_scores) / len(signal_scores)) if signal_scores else 0.0,
            "trigger_segment_count": len(self._trigger_segments),
            "trigger_end_reason_counts": dict(trigger_end_reason_counts),
            "left_hand_presence_ratio": left_hand_ratio,
            "right_hand_presence_ratio": right_hand_ratio,
            "pose_presence_ratio": pose_ratio,
            "max_invalid_streak": max_invalid_streak,
            "emits_per_min": (sum(emitted_label_counts.values()) / duration_minutes) if duration_minutes > 0 else 0.0,
            "no_sign_ratio": (
                stream_label_counts.get("待機中", 0) / max(len(self._rows), 1)
                if self._rows
                else 0.0
            ),
            "transition_like_ratio": (
                decoder_state_counts.get("transition_reject", 0) / max(len(self._rows), 1)
                if self._rows
                else 0.0
            ),
        }
        self.summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "session_dir": str(self.session_dir),
            "video_path": str(self.video_path),
            "frame_csv_path": str(self.frame_csv_path),
            "summary_json_path": str(self.summary_json_path),
        }


class RealtimeWordDecoder:
    BLANK_LABELS = {"no_sign", "unknown", "collecting", "transition"}
    BLANK_STATUSES = {"invalid_frame", "insufficient_signal", "collecting"}

    def __init__(
        self,
        *,
        alpha: float = 0.45,
        arm_frames: int = 4,
        release_frames: int = 5,
        min_confidence: float = 0.5,
        min_margin: float = 0.12,
        min_segment_frames: int = 6,
        cooldown_frames: int = 10,
        min_valid_ratio: float = 0.45,
        min_signal_score: float = 0.42,
        min_motion_energy: float = 0.015,
    ) -> None:
        self.alpha = alpha
        self.arm_frames = arm_frames
        self.release_frames = release_frames
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.min_segment_frames = min_segment_frames
        self.cooldown_frames = cooldown_frames
        self.min_valid_ratio = min_valid_ratio
        self.min_signal_score = min_signal_score
        self.min_motion_energy = min_motion_energy
        self.reset()

    def reset(self) -> None:
        self.ema_scores: dict[str, float] = {}
        self.active_count = 0
        self.inactive_count = 0
        self.segment_active = False
        self.segment_frames = 0
        self.segment_prob_sums: dict[str, float] = {}
        self.segment_profile_sums: dict[str, float] = {}
        self.segment_signal_sum = 0.0
        self.segment_valid_ratio_sum = 0.0
        self.segment_motion_max = 0.0
        self.last_emitted_label = ""
        self.last_emitted_confidence = 0.0
        self.cooldown_remaining = 0

    def describe(self) -> dict[str, object]:
        return {
            "kind": "segment_decoder",
            "alpha": self.alpha,
            "arm_frames": self.arm_frames,
            "release_frames": self.release_frames,
            "min_confidence": self.min_confidence,
            "min_margin": self.min_margin,
            "min_segment_frames": self.min_segment_frames,
            "cooldown_frames": self.cooldown_frames,
            "min_valid_ratio": self.min_valid_ratio,
            "min_signal_score": self.min_signal_score,
            "min_motion_energy": self.min_motion_energy,
        }

    def update(self, prediction) -> tuple[str, float, str | None, str]:
        emitted_label: str | None = None
        probs = prediction.assisted_probabilities or prediction.probabilities or {}
        self._update_ema(probs)
        best_label, best_score, best_margin = self._best_label()
        active = self._is_active(prediction, best_label, best_score, best_margin)

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if active:
            self.inactive_count = 0
            self.active_count += 1
            if not self.segment_active and self.active_count >= self.arm_frames:
                self._start_segment()
            if self.segment_active:
                self._consume_segment(prediction)
                label, score, _margin = self._segment_best()
                display_label = "收集中"
                return display_label, score, None, f"segment_active:{label or '-'}"
            return "收集中", best_score, None, "arming"

        self.active_count = 0
        if self.segment_active:
            self.inactive_count += 1
            if self.inactive_count >= self.release_frames:
                finalized = self._finalize_segment()
                return finalized
            label, score, _margin = self._segment_best()
            return "收集中", score, None, f"segment_hold:{label or '-'}"

        if self.cooldown_remaining > 0 and self.last_emitted_label:
            return self.last_emitted_label, self.last_emitted_confidence, None, "cooldown"
        return "待機中", 0.0, None, "idle"

    def _update_ema(self, probs: dict[str, float]) -> None:
        keys = set(self.ema_scores) | set(probs)
        if not keys:
            return
        for label in keys:
            previous = self.ema_scores.get(label, 0.0)
            current = probs.get(label, 0.0)
            self.ema_scores[label] = self.alpha * current + (1.0 - self.alpha) * previous

    def _best_label(self) -> tuple[str, float, float]:
        if not self.ema_scores:
            return "", 0.0, 0.0
        ranked = sorted(self.ema_scores.items(), key=lambda item: item[1], reverse=True)
        label, score = ranked[0]
        margin = float(score - ranked[1][1]) if len(ranked) >= 2 else float(score)
        return label, float(score), margin

    def _start_segment(self) -> None:
        self.segment_active = True
        self.segment_frames = 0
        self.segment_prob_sums = {}
        self.segment_profile_sums = {}
        self.segment_signal_sum = 0.0
        self.segment_valid_ratio_sum = 0.0
        self.segment_motion_max = 0.0

    def _consume_segment(self, prediction) -> None:
        self.segment_frames += 1
        for label, score in self.ema_scores.items():
            if label in self.BLANK_LABELS:
                continue
            self.segment_prob_sums[label] = self.segment_prob_sums.get(label, 0.0) + float(score)
        for label, score in (prediction.profile_probabilities or {}).items():
            if label in self.BLANK_LABELS:
                continue
            self.segment_profile_sums[label] = self.segment_profile_sums.get(label, 0.0) + float(score)
        self.segment_signal_sum += float(prediction.signal_score)
        self.segment_valid_ratio_sum += float(prediction.decision.valid_ratio)
        self.segment_motion_max = max(self.segment_motion_max, float(prediction.motion_energy))

    def _segment_best(self) -> tuple[str, float, float]:
        if self.segment_frames <= 0 or not self.segment_prob_sums:
            return "", 0.0, 0.0
        averaged_model = {
            label: score / self.segment_frames
            for label, score in self.segment_prob_sums.items()
        }
        averaged_profile = {
            label: score / self.segment_frames
            for label, score in self.segment_profile_sums.items()
        }
        labels = set(averaged_model) | set(averaged_profile)
        averaged = {
            label: (0.70 * averaged_model.get(label, 0.0)) + (0.30 * averaged_profile.get(label, 0.0))
            for label in labels
        }
        ranked = sorted(averaged.items(), key=lambda item: item[1], reverse=True)
        label, score = ranked[0]
        margin = float(score - ranked[1][1]) if len(ranked) >= 2 else float(score)
        return label, float(score), margin

    def _is_active(self, prediction, best_label: str, best_score: float, best_margin: float) -> bool:
        if prediction.decision.status in self.BLANK_STATUSES:
            return False
        has_hand = prediction.left_hand_present or prediction.right_hand_present
        if not has_hand or not prediction.pose_present:
            return False
        if prediction.signal_score >= self.min_signal_score and prediction.motion_energy >= self.min_motion_energy:
            return True
        return (
            best_label not in self.BLANK_LABELS
            and best_score >= max(self.min_confidence, 0.55)
            and best_margin >= self.min_margin
            and prediction.decision.valid_ratio >= self.min_valid_ratio
            and prediction.signal_score >= (self.min_signal_score - 0.05)
        )

    def _finalize_segment(self) -> tuple[str, float, str | None, str]:
        label, score, margin = self._segment_best()
        segment_frames = self.segment_frames
        avg_signal = self.segment_signal_sum / max(segment_frames, 1)
        avg_valid_ratio = self.segment_valid_ratio_sum / max(segment_frames, 1)
        motion_peak = self.segment_motion_max
        self.segment_active = False
        self.inactive_count = 0
        self.segment_frames = 0
        self.segment_prob_sums = {}
        self.segment_profile_sums = {}
        self.segment_signal_sum = 0.0
        self.segment_valid_ratio_sum = 0.0
        self.segment_motion_max = 0.0

        if (
            not label
            or segment_frames < self.min_segment_frames
            or score < self.min_confidence
            or margin < self.min_margin
            or avg_signal < self.min_signal_score
            or (avg_valid_ratio < self.min_valid_ratio and motion_peak < self.min_motion_energy)
        ):
            return "過渡中", score, None, "transition_reject"

        if label == self.last_emitted_label and self.cooldown_remaining > 0:
            return label, score, None, "duplicate_reject"

        self.last_emitted_label = label
        self.last_emitted_confidence = score
        self.cooldown_remaining = self.cooldown_frames
        return label, score, label, "emit"


class SentenceInterfaceApp:
    DEFAULT_MODE_PRESETS = {
        "句子 30 類模式": {
            "checkpoint": r"c:\Users\qqq02\Desktop\99_docs_analysis\integration_workspace\artifacts_30_sentence_boost_seq40s2_run2\models\multibranch_baseline.pt",
            "examples": r"c:\Users\qqq02\Desktop\99_docs_analysis\metadata\asl_training_answers.csv",
            "gesture_profile": "",
            "sequence_length": 40,
            "confidence": 0.55,
            "mirror_input": False,
            "notes": "30 類句子辨識模式",
        },
        "webcam 9 類即時模式": {
            "checkpoint": r"c:\Users\qqq02\Desktop\99_docs_analysis\integration_workspace\artifacts_webcam9_nosign_seq30s5_iso\models\multibranch_baseline.pt",
            "examples": r"c:\Users\qqq02\Desktop\99_docs_analysis\metadata\asl_training_answers.csv",
            "gesture_profile": r"c:\Users\qqq02\Desktop\99_docs_analysis\metadata\webcam9_gesture_profiles.json",
            "sequence_length": 30,
            "confidence": 0.35,
            "mirror_input": True,
            "notes": "9 類 webcam 即時模式，包含 no_sign，使用零重疊 source_group_key split、train-only gesture profile、segment decoder。",
        },
    }
    DEFAULT_DECODER_PRESETS = {
        "default": {
            "alpha": 0.45,
            "arm_frames": 4,
            "release_frames": 5,
            "min_confidence": 0.45,
            "min_margin": 0.12,
            "min_segment_frames": 6,
            "cooldown_frames": 10,
            "min_valid_ratio": 0.45,
            "min_signal_score": 0.42,
            "min_motion_energy": 0.015,
        },
        "webcam 9 類即時模式": {
            "min_confidence": 0.45,
            "min_margin": 0.12,
            "arm_frames": 4,
            "release_frames": 5,
            "min_segment_frames": 6,
            "cooldown_frames": 10,
            "min_valid_ratio": 0.45,
            "min_signal_score": 0.42,
            "min_motion_energy": 0.015,
        },
    }
    TEST_LOG_ROOT = Path(r"c:\Users\qqq02\Desktop\99_docs_analysis\reports\realtime_tests")

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("手語句子介面")
        self.root.geometry("920x640")

        preset_overrides, decoder_overrides, notices = load_user_tuning_settings()
        self.mode_presets = _deep_update(self.DEFAULT_MODE_PRESETS, preset_overrides)
        self.decoder_presets = _deep_update(self.DEFAULT_DECODER_PRESETS, decoder_overrides)

        self.mode_var = tk.StringVar(value="句子 30 類模式")
        self.checkpoint_var = tk.StringVar(
            value=str(Path(self.mode_presets["句子 30 類模式"]["checkpoint"]))
        )
        self.examples_var = tk.StringVar(
            value=str(Path(self.mode_presets["句子 30 類模式"]["examples"]))
        )
        self.gesture_profile_var = tk.StringVar(
            value=str(self.mode_presets["句子 30 類模式"].get("gesture_profile", ""))
        )
        self.sequence_length_var = tk.IntVar(value=int(self.mode_presets["句子 30 類模式"]["sequence_length"]))
        self.confidence_var = tk.DoubleVar(value=float(self.mode_presets["句子 30 類模式"]["confidence"]))
        self.camera_index_var = tk.IntVar(value=0)
        self.mirror_input_var = tk.BooleanVar(value=bool(self.mode_presets["句子 30 類模式"]["mirror_input"]))
        self.video_path_var = tk.StringVar(value="")
        self.batch_summary_var = tk.StringVar(value="尚未執行批次影片")
        initial_status = notices[-1] if notices else "待命中"
        self.status_var = tk.StringVar(value=initial_status)
        self.current_prediction_var = tk.StringVar(value="尚未辨識")
        self.current_gloss_var = tk.StringVar(value="尚未組合")
        self.current_english_var = tk.StringVar(value="尚未組合")
        self.current_chinese_var = tk.StringVar(value="尚未組合")
        self.current_notes_var = tk.StringVar(value="等待辨識結果")
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.collector = PhraseCollector(min_gap_frames=24)
        self.word_decoder = RealtimeWordDecoder(**self._get_decoder_settings())
        self.composer = SentenceComposer(self.examples_var.get())
        self.settings_window: tk.Toplevel | None = None
        self.log_visible = True
        self.video_queue: list[str] = []
        self.batch_results: list[dict[str, str]] = []

        self._build_layout()
        self.root.after(100, self._drain_queue)

    def _build_layout(self) -> None:
        top_bar = ttk.Frame(self.root, padding=12)
        top_bar.pack(fill=tk.X)
        ttk.Label(top_bar, text="模式").pack(side=tk.LEFT, padx=(0, 6))
        self.mode_combo = ttk.Combobox(
            top_bar,
            textvariable=self.mode_var,
            values=list(self.mode_presets.keys()),
            state="readonly",
            width=20,
        )
        self.mode_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)
        ttk.Button(top_bar, text="即時辨識", command=self.start_realtime).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top_bar, text="選擇影片", command=self._browse_video).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top_bar, text="多選影片", command=self._browse_multiple_videos).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top_bar, text="執行影片", command=self.run_video).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top_bar, text="影片模擬即時", command=self.run_video_as_realtime).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top_bar, text="執行全部影片", command=self.run_video_batch).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top_bar, text="組合句子", command=self.compose_sentence).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top_bar, text="清除內容", command=self.clear_tokens).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top_bar, text="停止", command=self.stop_worker).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top_bar, text="顯示/隱藏紀錄", command=self.toggle_log_panel).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top_bar, text="設定", command=self.open_settings_window).pack(side=tk.RIGHT)

        video_bar = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        video_bar.pack(fill=tk.X)
        ttk.Label(video_bar, text="影片檔案").pack(side=tk.LEFT)
        ttk.Entry(video_bar, textvariable=self.video_path_var, width=90).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        queue_frame = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        queue_frame.pack(fill=tk.X)
        ttk.Label(queue_frame, text="影片清單").pack(anchor="w")
        queue_body = ttk.Frame(queue_frame)
        queue_body.pack(fill=tk.X)
        self.video_listbox = tk.Listbox(queue_body, height=4)
        self.video_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        queue_actions = ttk.Frame(queue_body)
        queue_actions.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(queue_actions, text="加入目前影片", command=self.add_current_video_to_queue).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(queue_actions, text="移除選取", command=self.remove_selected_videos).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(queue_actions, text="清空清單", command=self.clear_video_queue).pack(fill=tk.X)

        hero_frame = ttk.Frame(self.root, padding=(12, 0, 12, 14))
        hero_frame.pack(fill=tk.X)
        hero_card = ttk.LabelFrame(hero_frame, text="即時字幕", padding=14)
        hero_card.pack(fill=tk.X)
        self.subtitle_label = tk.Label(
            hero_card,
            textvariable=self.current_prediction_var,
            font=("Microsoft JhengHei UI", 24, "bold"),
            anchor="w",
            justify="left",
            bg="#f7f7f2",
            fg="#1f2937",
            padx=12,
            pady=14,
        )
        self.subtitle_label.pack(fill=tk.X)

        status_frame = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, text="狀態").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=6)

        panes = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        left = ttk.Frame(panes, padding=8)
        right = ttk.Frame(panes, padding=8)
        panes.add(left, weight=1)
        panes.add(right, weight=1)

        ttk.Label(left, text="辨識詞語").pack(anchor="w")
        self.tokens_box = tk.Text(left, height=12, wrap="word")
        self.tokens_box.pack(fill=tk.BOTH, expand=True)

        self.log_header = ttk.Label(left, text="辨識紀錄")
        self.log_header.pack(anchor="w", pady=(8, 0))
        self.log_box = tk.Text(left, height=20, wrap="word")
        self.log_box.pack(fill=tk.BOTH, expand=True)

        result_card = ttk.LabelFrame(right, text="句子結果", padding=12)
        result_card.pack(fill=tk.BOTH, expand=True)
        self._build_result_card(result_card)
        ttk.Label(right, text="批次摘要").pack(anchor="w", pady=(10, 0))
        self.batch_summary_label = tk.Label(
            right,
            textvariable=self.batch_summary_var,
            font=("Microsoft JhengHei UI", 11),
            anchor="w",
            justify="left",
            wraplength=380,
            bg="#fff8e7",
            fg="#3f2f16",
            padx=12,
            pady=10,
        )
        self.batch_summary_label.pack(fill=tk.X, pady=(4, 0))

    def _build_result_card(self, parent: ttk.LabelFrame) -> None:
        self._make_result_row(parent, "Gloss", self.current_gloss_var, ("Consolas", 16, "bold"))
        self._make_result_row(parent, "英文", self.current_english_var, ("Microsoft JhengHei UI", 14, "bold"))
        self._make_result_row(parent, "中文", self.current_chinese_var, ("Microsoft JhengHei UI", 16, "bold"))
        self._make_result_row(parent, "說明", self.current_notes_var, ("Microsoft JhengHei UI", 11))

    def _make_result_row(self, parent, title: str, variable: tk.StringVar, font) -> None:
        box = ttk.Frame(parent, padding=(0, 0, 0, 10))
        box.pack(fill=tk.X, anchor="n")
        ttk.Label(box, text=title).pack(anchor="w")
        label = tk.Label(
            box,
            textvariable=variable,
            font=font,
            anchor="w",
            justify="left",
            wraplength=380,
            bg="#f8fafc",
            fg="#111827",
            padx=12,
            pady=10,
        )
        label.pack(fill=tk.X, pady=(4, 0))

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇影片檔",
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("所有檔案", "*.*")],
        )
        if path:
            self.video_path_var.set(path)

    def _browse_multiple_videos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="選擇多個影片檔",
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("所有檔案", "*.*")],
        )
        if not paths:
            return
        for path in paths:
            if path not in self.video_queue:
                self.video_queue.append(path)
        self.video_path_var.set(paths[0])
        self._refresh_video_queue()
        self.status_var.set(f"已加入 {len(paths)} 部影片")

    def add_current_video_to_queue(self) -> None:
        path = self.video_path_var.get().strip()
        if not path:
            messagebox.showerror("缺少影片", "請先選擇影片檔。")
            return
        if path not in self.video_queue:
            self.video_queue.append(path)
        self._refresh_video_queue()
        self.status_var.set("已加入影片清單")

    def remove_selected_videos(self) -> None:
        selected = list(self.video_listbox.curselection())
        if not selected:
            return
        for index in reversed(selected):
            del self.video_queue[index]
        self._refresh_video_queue()
        self.status_var.set("已移除選取影片")

    def clear_video_queue(self) -> None:
        self.video_queue.clear()
        self._refresh_video_queue()
        self.batch_summary_var.set("尚未執行批次影片")
        self.status_var.set("已清空影片清單")

    def _refresh_video_queue(self) -> None:
        self.video_listbox.delete(0, tk.END)
        for index, path in enumerate(self.video_queue, start=1):
            self.video_listbox.insert(tk.END, f"{index}. {Path(path).name}")

    def clear_tokens(self) -> None:
        self.collector.clear()
        self._reset_word_decoder()
        self.tokens_box.delete("1.0", tk.END)
        self.log_box.delete("1.0", tk.END)
        self.current_prediction_var.set("尚未辨識")
        self.current_gloss_var.set("尚未組合")
        self.current_english_var.set("尚未組合")
        self.current_chinese_var.set("尚未組合")
        self.current_notes_var.set("等待辨識結果")
        self.batch_summary_var.set("尚未執行批次影片")
        self.status_var.set("已清除內容")

    def _reset_word_decoder(self) -> None:
        self.word_decoder = RealtimeWordDecoder(**self._get_decoder_settings())

    def _get_decoder_settings(self) -> dict[str, object]:
        return get_decoder_settings_for_mode(
            self.mode_var.get(),
            confidence_threshold=float(self.confidence_var.get()),
        )

    def _is_webcam_test_mode(self) -> bool:
        return self.mode_var.get() == "webcam 9 類即時模式"

    def compose_sentence(self) -> None:
        if self._is_webcam_test_mode():
            self.current_gloss_var.set(self.current_prediction_var.get())
            self.current_english_var.set(self.current_prediction_var.get())
            self.current_chinese_var.set("webcam 9 類即時測試")
            self.current_notes_var.set("9 類即時模式不進行句子組裝。")
            self.status_var.set("已更新單字測試結果")
            return
        self.composer = SentenceComposer(self.examples_var.get())
        result = self.composer.compose(self.collector.tokens)
        self.current_gloss_var.set(result.gloss or "尚未組合")
        self.current_english_var.set(result.english or "尚未組合")
        self.current_chinese_var.set(result.chinese or "尚未組合")
        source_text = self._source_text(result.source)
        self.current_notes_var.set(f"{source_text}：{result.notes}")
        self.status_var.set("已完成句子組合")

    def _append_log(self, text: str) -> None:
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)

    def _refresh_tokens(self) -> None:
        self.tokens_box.delete("1.0", tk.END)
        if self._is_webcam_test_mode():
            return
        normalized_tokens = self.composer.normalize_tokens(self.collector.tokens)
        pretty = "  ".join(word.upper() for word in normalized_tokens)
        self.tokens_box.insert(tk.END, pretty)

    def toggle_log_panel(self) -> None:
        if self.log_visible:
            self.log_header.pack_forget()
            self.log_box.pack_forget()
            self.log_visible = False
            self.status_var.set("已隱藏辨識紀錄")
            return
        self.log_header.pack(anchor="w", pady=(8, 0))
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_visible = True
        self.status_var.set("已顯示辨識紀錄")

    def start_realtime(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("忙碌中", "目前已有作業執行中。")
            return
        self.stop_event.clear()
        self._reset_word_decoder()
        self.worker_thread = threading.Thread(target=self._run_realtime_worker, daemon=True)
        self.worker_thread.start()
        self.status_var.set("即時辨識啟動中")

    def run_video(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("忙碌中", "目前已有作業執行中。")
            return
        if not self.video_path_var.get().strip():
            messagebox.showerror("缺少影片", "請先選擇影片檔。")
            return
        self.stop_event.clear()
        self._reset_word_decoder()
        self.worker_thread = threading.Thread(target=self._run_video_worker, daemon=True)
        self.worker_thread.start()
        self.status_var.set("影片辨識啟動中")

    def run_video_as_realtime(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("忙碌中", "目前已有作業執行中。")
            return
        if not self.video_path_var.get().strip():
            messagebox.showerror("缺少影片", "請先選擇影片檔。")
            return
        self.stop_event.clear()
        self._reset_current_result_state(clear_log=False)
        self._reset_word_decoder()
        self.worker_thread = threading.Thread(target=self._run_video_realtime_worker, daemon=True)
        self.worker_thread.start()
        self.status_var.set("影片模擬即時啟動中")

    def run_video_batch(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("忙碌中", "目前已有作業執行中。")
            return
        if not self.video_queue:
            messagebox.showerror("缺少影片", "請先加入至少一部影片到清單。")
            return
        self.stop_event.clear()
        self.batch_results = []
        self.worker_thread = threading.Thread(target=self._run_video_batch_worker, daemon=True)
        self.worker_thread.start()
        self.status_var.set(f"批次影片啟動中，共 {len(self.video_queue)} 部")

    def stop_worker(self) -> None:
        self.stop_event.set()
        self.status_var.set("停止中")

    def open_settings_window(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        window = tk.Toplevel(self.root)
        window.title("設定")
        window.geometry("760x220")
        window.resizable(False, False)
        self.settings_window = window

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="預設模式").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self.mode_var,
            values=list(self.mode_presets.keys()),
            state="readonly",
            width=30,
        ).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(frame, text="模型檔案").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.checkpoint_var, width=80).grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(frame, text="選擇", command=self._browse_checkpoint).grid(row=1, column=2, sticky="w")

        ttk.Label(frame, text="語句範例 CSV").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.examples_var, width=80).grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(frame, text="選擇", command=self._browse_examples_csv).grid(row=2, column=2, sticky="w")

        ttk.Label(frame, text="手勢範圍 JSON").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.gesture_profile_var, width=80).grid(row=3, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(frame, text="選擇", command=self._browse_gesture_profile).grid(row=3, column=2, sticky="w")

        ttk.Label(frame, text="相機索引").grid(row=4, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.camera_index_var, width=10).grid(row=4, column=1, sticky="w", padx=6, pady=4)

        ttk.Checkbutton(frame, text="即時輸入鏡像", variable=self.mirror_input_var).grid(row=5, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(frame, text="序列長度").grid(row=6, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.sequence_length_var, width=10).grid(row=6, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(frame, text="信心門檻").grid(row=7, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.confidence_var, width=10).grid(row=7, column=1, sticky="w", padx=6, pady=4)

        ttk.Button(frame, text="套用", command=self._apply_settings).grid(row=8, column=1, sticky="e", pady=(10, 0))
        frame.columnconfigure(1, weight=1)

    def _on_mode_changed(self, _event=None) -> None:
        self._apply_mode_preset(update_status=True)

    def _apply_mode_preset(self, update_status: bool = False) -> None:
        preset = self.mode_presets.get(self.mode_var.get())
        if preset is None:
            return
        self.checkpoint_var.set(str(Path(str(preset["checkpoint"]))))
        self.examples_var.set(str(Path(str(preset["examples"]))))
        self.gesture_profile_var.set(str(preset.get("gesture_profile", "")))
        self.sequence_length_var.set(int(preset["sequence_length"]))
        self.confidence_var.set(float(preset["confidence"]))
        self.mirror_input_var.set(bool(preset.get("mirror_input", False)))
        if update_status:
            self.clear_tokens()
            self.current_notes_var.set(str(preset["notes"]))
            self.status_var.set(f"已切換為 {self.mode_var.get()}")

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇模型檔案",
            filetypes=[("PyTorch 模型", "*.pt"), ("所有檔案", "*.*")],
        )
        if path:
            self.checkpoint_var.set(path)

    def _browse_examples_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇語句範例 CSV",
            filetypes=[("CSV 檔案", "*.csv"), ("所有檔案", "*.*")],
        )
        if path:
            self.examples_var.set(path)

    def _browse_gesture_profile(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇手勢範圍 JSON",
            filetypes=[("JSON 檔案", "*.json"), ("所有檔案", "*.*")],
        )
        if path:
            self.gesture_profile_var.set(path)

    def _apply_settings(self) -> None:
        self._apply_mode_preset(update_status=False)
        self.composer = SentenceComposer(self.examples_var.get())
        self.status_var.set("設定已套用")
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.destroy()
            self.settings_window = None

    def _build_engine(self) -> MultibranchSequenceEngine:
        return MultibranchSequenceEngine(
            checkpoint_path=self.checkpoint_var.get().strip(),
            sequence_length=int(self.sequence_length_var.get()),
            confidence_threshold=float(self.confidence_var.get()),
            gesture_profile_path=self.gesture_profile_var.get().strip() or None,
        )

    def _run_realtime_worker(self) -> None:
        engine = None
        capture = None
        recorder = None
        try:
            engine = self._build_engine()
            capture = cv2.VideoCapture(int(self.camera_index_var.get()))
            if not capture.isOpened():
                raise RuntimeError("無法開啟即時相機。")
            recorder = RealtimeTestRecorder(
                output_root=self.TEST_LOG_ROOT,
                mode_name=self.mode_var.get(),
                checkpoint_path=self.checkpoint_var.get().strip(),
                confidence_threshold=float(self.confidence_var.get()),
                sequence_length=int(self.sequence_length_var.get()),
                mirror_input=bool(self.mirror_input_var.get()),
                camera_index=int(self.camera_index_var.get()),
                gesture_profile_path=self.gesture_profile_var.get().strip(),
                decoder_params=self.word_decoder.describe(),
            )
            self.ui_queue.put(("status", f"即時辨識啟動中，已開始記錄：{recorder.session_dir}"))
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("無法讀取即時畫面。")
                if self.mirror_input_var.get():
                    frame = cv2.flip(frame, 1)
                prediction = engine.process_frame(frame)
                stream_payload = self._handle_prediction(prediction, preview_name="即時手語預覽")
                recorder.write_frame(
                    prediction.display_frame,
                    frame_index=prediction.frame_index,
                    raw_label=prediction.decision.raw_label,
                    raw_confidence=prediction.decision.raw_confidence,
                    final_label=prediction.decision.final_label,
                    final_confidence=prediction.decision.final_confidence,
                    status=prediction.decision.status,
                    valid_ratio=prediction.decision.valid_ratio,
                    top_candidates=prediction.top_candidates,
                    stream_label=stream_payload["stream_label"] if stream_payload else "",
                    stream_confidence=stream_payload["stream_confidence"] if stream_payload else 0.0,
                    decoder_state=stream_payload["decoder_state"] if stream_payload else "",
                    emitted_label=stream_payload["emitted_label"] if stream_payload else "",
                    assisted_top_candidates=stream_payload["assisted_top_candidates"] if stream_payload else [],
                    assist_notes=stream_payload["assist_notes"] if stream_payload else "",
                    probabilities=prediction.probabilities,
                    assisted_probabilities=prediction.assisted_probabilities,
                    disambiguation_notes=prediction.disambiguation_notes,
                    disambiguation_applied=prediction.disambiguation_applied,
                    motion_energy=prediction.motion_energy,
                    top_margin=prediction.top_margin,
                    signal_score=prediction.signal_score,
                    left_hand_present=prediction.left_hand_present,
                    right_hand_present=prediction.right_hand_present,
                    pose_present=prediction.pose_present,
                    trigger_segment_debug=prediction.trigger_segment_debug,
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as exc:
            self.ui_queue.put(("status", f"即時辨識錯誤：{exc}"))
        finally:
            if capture is not None:
                capture.release()
            cv2.destroyAllWindows()
            if recorder is not None:
                saved = recorder.close()
                self.ui_queue.put(
                    (
                        "status",
                        f"即時辨識已結束，記錄已保存到：{saved['session_dir']}",
                    )
                )
            if engine is not None:
                engine.close()

    def _run_video_worker(self) -> None:
        engine = None
        capture = None
        try:
            engine = self._build_engine()
            video_path = self.video_path_var.get().strip()
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                raise RuntimeError(f"無法開啟影片：{video_path}")
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                prediction = engine.process_frame(frame)
                self._handle_prediction(prediction, preview_name="影片手語預覽")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            flushed = self.collector.flush(max(engine.frame_index - 1, 0))
            if flushed is not None:
                self.ui_queue.put(("tokens", None))
            self.ui_queue.put(("compose", None))
        except Exception as exc:
            self.ui_queue.put(("status", f"影片辨識錯誤：{exc}"))
        finally:
            if capture is not None:
                capture.release()
            cv2.destroyAllWindows()
            if engine is not None:
                engine.close()
            self.ui_queue.put(("status", "影片辨識已結束"))

    def _run_video_realtime_worker(self) -> None:
        engine = None
        capture = None
        recorder = None
        scripted_frame_signals: list[ScriptedFrameSignal] = []
        try:
            engine = self._build_engine()
            video_path = self.video_path_var.get().strip()
            scripted_tokens = parse_script_tokens_from_path(video_path, allowed_labels=set(engine.index_to_label.values()))
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                raise RuntimeError(f"無法開啟影片：{video_path}")
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            playback_fps = source_fps if source_fps > 1.0 else 20.0
            recorder = RealtimeTestRecorder(
                output_root=self.TEST_LOG_ROOT,
                mode_name=f"{self.mode_var.get()} / 影片模擬即時",
                checkpoint_path=self.checkpoint_var.get().strip(),
                confidence_threshold=float(self.confidence_var.get()),
                sequence_length=int(self.sequence_length_var.get()),
                mirror_input=bool(self.mirror_input_var.get()),
                camera_index=-1,
                gesture_profile_path=self.gesture_profile_var.get().strip(),
                decoder_params={**self.word_decoder.describe(), "playback_fps": playback_fps},
                source_video_path=video_path,
            )
            if scripted_tokens:
                self.ui_queue.put(("current", "影片模擬即時執行中"))
                self.ui_queue.put(
                    (
                        "status",
                        f"影片模擬即時啟動中，已開始記錄：{recorder.session_dir}。scripted 對齊結果會在影片結束後顯示。",
                    )
                )
            else:
                self.ui_queue.put(("status", f"影片模擬即時啟動中，已開始記錄：{recorder.session_dir}"))
            frame_interval = 1.0 / max(playback_fps, 1.0)
            while not self.stop_event.is_set():
                started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                if self.mirror_input_var.get():
                    frame = cv2.flip(frame, 1)
                prediction = engine.process_frame(frame)
                stream_payload = self._handle_prediction(prediction, preview_name="影片模擬即時預覽")
                scripted_frame_signals.append(
                    ScriptedFrameSignal(
                        frame_index=prediction.frame_index,
                        signal_score=prediction.signal_score,
                        motion_energy=prediction.motion_energy,
                        status=prediction.decision.status,
                        left_hand_present=prediction.left_hand_present,
                        right_hand_present=prediction.right_hand_present,
                        pose_present=prediction.pose_present,
                    )
                )
                recorder.write_frame(
                    prediction.display_frame,
                    frame_index=prediction.frame_index,
                    raw_label=prediction.decision.raw_label,
                    raw_confidence=prediction.decision.raw_confidence,
                    final_label=prediction.decision.final_label,
                    final_confidence=prediction.decision.final_confidence,
                    status=prediction.decision.status,
                    valid_ratio=prediction.decision.valid_ratio,
                    top_candidates=prediction.top_candidates,
                    stream_label=stream_payload["stream_label"] if stream_payload else "",
                    stream_confidence=stream_payload["stream_confidence"] if stream_payload else 0.0,
                    decoder_state=stream_payload["decoder_state"] if stream_payload else "",
                    emitted_label=stream_payload["emitted_label"] if stream_payload else "",
                    assisted_top_candidates=stream_payload["assisted_top_candidates"] if stream_payload else [],
                    assist_notes=stream_payload["assist_notes"] if stream_payload else "",
                    probabilities=prediction.probabilities,
                    assisted_probabilities=prediction.assisted_probabilities,
                    disambiguation_notes=prediction.disambiguation_notes,
                    disambiguation_applied=prediction.disambiguation_applied,
                    motion_energy=prediction.motion_energy,
                    top_margin=prediction.top_margin,
                    signal_score=prediction.signal_score,
                    left_hand_present=prediction.left_hand_present,
                    right_hand_present=prediction.right_hand_present,
                    pose_present=prediction.pose_present,
                    trigger_segment_debug=prediction.trigger_segment_debug,
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                elapsed = time.perf_counter() - started
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
            scripted_alignment = build_scripted_alignment(
                video_path,
                scripted_frame_signals,
                allowed_labels=set(engine.index_to_label.values()),
            )
            if recorder is not None and scripted_alignment is not None:
                alignment_path = Path(recorder.session_dir) / "scripted_alignment.json"
                alignment_path.write_text(
                    json.dumps(scripted_alignment, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tokens = scripted_alignment.get("tokens", [])
                if tokens:
                    self.ui_queue.put(("scripted_alignment", scripted_alignment))
                    self.ui_queue.put(
                        (
                            "status",
                            f"影片模擬即時已完成 scripted 對齊：{' / '.join(tokens)}",
                        )
                    )
        except Exception as exc:
            self.ui_queue.put(("status", f"影片模擬即時錯誤：{exc}"))
        finally:
            if capture is not None:
                capture.release()
            cv2.destroyAllWindows()
            if recorder is not None:
                saved = recorder.close()
                self.ui_queue.put(("status", f"影片模擬即時已結束，記錄已保存到：{saved['session_dir']}"))
            if engine is not None:
                engine.close()

    def _run_video_batch_worker(self) -> None:
        try:
            total = len(self.video_queue)
            for index, video_path in enumerate(self.video_queue, start=1):
                if self.stop_event.is_set():
                    break
                self.ui_queue.put(("status", f"批次處理中 {index}/{total}：{Path(video_path).name}"))
                self._reset_current_result_state(clear_log=False)
                self._process_single_video(video_path)
                composition = self.composer.compose(self.collector.tokens)
                self.batch_results.append(
                    {
                        "name": Path(video_path).name,
                        "gloss": composition.gloss or "-",
                        "english": composition.english or "-",
                    }
                )
                self.ui_queue.put(("composition", composition))
                self.ui_queue.put(("batch_summary", None))
            self.ui_queue.put(("status", "批次影片已完成"))
        except Exception as exc:
            self.ui_queue.put(("status", f"批次影片錯誤：{exc}"))
        finally:
            cv2.destroyAllWindows()

    def _process_single_video(self, video_path: str) -> None:
        engine = self._build_engine()
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            engine.close()
            raise RuntimeError(f"無法開啟影片：{video_path}")
        try:
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                prediction = engine.process_frame(frame)
                self._handle_prediction(prediction, preview_name="影片手語預覽")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            flushed = self.collector.flush(max(engine.frame_index - 1, 0))
            if flushed is not None:
                self.ui_queue.put(("tokens", None))
        finally:
            capture.release()
            engine.close()

    def _reset_current_result_state(self, clear_log: bool) -> None:
        self.collector.clear()
        self._reset_word_decoder()
        self.current_prediction_var.set("尚未辨識")
        self.current_gloss_var.set("尚未組合")
        self.current_english_var.set("尚未組合")
        self.current_chinese_var.set("尚未組合")
        self.current_notes_var.set("等待辨識結果")
        self.tokens_box.delete("1.0", tk.END)
        if clear_log:
            self.log_box.delete("1.0", tk.END)

    def _refresh_batch_summary(self) -> None:
        if not self.batch_results:
            self.batch_summary_var.set("尚未執行批次影片")
            return
        lines = []
        for index, item in enumerate(self.batch_results[-5:], start=max(len(self.batch_results) - 4, 1)):
            lines.append(f"{index}. {item['name']} | {item['gloss']} | {item['english']}")
        self.batch_summary_var.set("\n".join(lines))

    def _handle_prediction(self, prediction, preview_name: str) -> dict[str, object] | None:
        cv2.imshow(preview_name, prediction.display_frame)
        if self._is_webcam_test_mode():
            if getattr(prediction, "engine_mode", "sliding_window") == "trigger_based":
                emitted_label = prediction.emitted_label or None
                if emitted_label:
                    stream_label = emitted_label
                    stream_confidence = float(prediction.decision.final_confidence)
                    decoder_state = prediction.decision.status
                else:
                    stream_label = "收集中" if prediction.decision.status == "trigger_collecting" else "待機中"
                    stream_confidence = 0.0
                    decoder_state = prediction.decision.status
            else:
                stream_label, stream_confidence, emitted_label, decoder_state = self.word_decoder.update(prediction)
            raw_label = prediction.decision.raw_label
            confidence = prediction.decision.raw_confidence
            self.ui_queue.put(
                (
                    "prediction",
                    f"{prediction.frame_index}: raw={raw_label} ({confidence:.2f}) final={prediction.decision.final_label} "
                    f"stream={stream_label} ({stream_confidence:.2f}) "
                    f"[{prediction.decision.status}/{decoder_state}] motion={prediction.motion_energy:.3f} "
                    f"margin={prediction.top_margin:.2f} signal={prediction.signal_score:.2f}",
                )
            )
            self.ui_queue.put(("current", stream_label))
            self.ui_queue.put(
                (
                    "webcam_test_result",
                    (
                        stream_label,
                        stream_confidence,
                        emitted_label,
                        decoder_state,
                        raw_label,
                        confidence,
                        prediction.top_candidates,
                        prediction.assisted_top_candidates,
                        prediction.assist_notes,
                        prediction.disambiguation_notes,
                        prediction.motion_energy,
                        prediction.top_margin,
                        prediction.signal_score,
                    ),
                )
            )
            return {
                "stream_label": stream_label,
                "stream_confidence": stream_confidence,
                "decoder_state": decoder_state,
                "emitted_label": emitted_label or "",
                "assisted_top_candidates": prediction.assisted_top_candidates,
                "assist_notes": prediction.assist_notes,
            }

        final_label = prediction.decision.final_label
        display_label = "收集中" if final_label == "collecting" else final_label
        self.ui_queue.put(("prediction", f"{prediction.frame_index}: {display_label} [{prediction.decision.status}]"))
        self.ui_queue.put(("current", display_label))
        event = self.collector.update(final_label, prediction.frame_index, prediction.decision.final_confidence)
        if event is not None:
            self.ui_queue.put(("tokens", None))
        return None

    def _drain_queue(self) -> None:
        while True:
            try:
                event, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if event == "status":
                self.status_var.set(str(payload))
            elif event == "prediction":
                self._append_log(str(payload))
            elif event == "current":
                self.current_prediction_var.set(str(payload))
            elif event == "tokens":
                self._refresh_tokens()
            elif event == "compose":
                self.compose_sentence()
            elif event == "webcam_test_result":
                (
                    label,
                    confidence,
                    emitted_label,
                    decoder_state,
                    raw_label,
                    raw_confidence,
                    top_candidates,
                    assisted_top_candidates,
                    assist_notes,
                    disambiguation_notes,
                    motion_energy,
                    top_margin,
                    signal_score,
                ) = payload
                top3_text = " / ".join(
                    f"{item_label}:{item_conf:.2f}" for item_label, item_conf in assisted_top_candidates[:3]
                )
                if emitted_label:
                    self.current_gloss_var.set(emitted_label.upper())
                    self.current_english_var.set(emitted_label)
                    self.current_chinese_var.set("webcam 9 類即時測試")
                elif label == "待機中":
                    self.current_gloss_var.set("NO_SIGN")
                    self.current_english_var.set("no_sign")
                    self.current_chinese_var.set("無手勢")
                elif label == "過渡中":
                    self.current_gloss_var.set("TRANSITION")
                    self.current_english_var.set("transition")
                    self.current_chinese_var.set("過渡動作")
                self.current_notes_var.set(
                    f"stream={label}({confidence:.2f})，raw={raw_label}({raw_confidence:.2f})，state={decoder_state}，"
                    f"motion={motion_energy:.3f}，margin={top_margin:.2f}，signal={signal_score:.2f}，"
                    f"Assist={top3_text}，{assist_notes}，{disambiguation_notes}"
                )
            elif event == "composition":
                result = payload
                self.current_gloss_var.set(result.gloss or "尚未組合")
                self.current_english_var.set(result.english or "尚未組合")
                self.current_chinese_var.set(result.chinese or "尚未組合")
                source_text = self._source_text(result.source)
                self.current_notes_var.set(f"{source_text}：{result.notes}")
                self._refresh_tokens()
            elif event == "scripted_alignment":
                alignment = payload
                tokens = list(alignment.get("tokens", []))
                if tokens:
                    self.current_prediction_var.set(" / ".join(tokens))
                    self.current_gloss_var.set(" ".join(token.upper() for token in tokens))
                    self.current_english_var.set(" ".join(tokens))
                    self.current_chinese_var.set("scripted 測試影片對齊")
                    self.current_notes_var.set(
                        f"段數 {alignment.get('aligned_segment_count', 0)}/{len(alignment.get('expected_tokens', []))}，"
                        f"原始偵測段數 {alignment.get('detected_segment_count', 0)}"
                    )
                    self.tokens_box.delete("1.0", tk.END)
                    self.tokens_box.insert(tk.END, "  ".join(tokens))
            elif event == "batch_summary":
                self._refresh_batch_summary()
        self.root.after(100, self._drain_queue)

    def _source_text(self, source: str) -> str:
        if source == "llm":
            return "語言模型"
        if source == "template":
            return "模板解碼"
        return "規則引擎"


def main() -> None:
    root = tk.Tk()
    app = SentenceInterfaceApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_worker(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
