from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.sign_sentence_engine import MultibranchSequenceEngine, TriggerFrameRecord
from src.dataset.continuous_feature_cache import load_continuous_feature_cache
from src.landmarks.holistic_extractor import FrameLandmarks


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Analyze the serving-path you-vs-like scorer margin on the target trigger segment.")
    parser.add_argument("--baseline-session", required=True)
    parser.add_argument("--current-session", required=True)
    parser.add_argument("--exact-classification-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gesture-profile", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: float) -> float:
    return round(float(value), 6)


def _read_alignment_row(session_dir: Path, reference_label: str) -> dict[str, str]:
    alignment_path = session_dir / "trigger_segment_alignment.csv"
    with alignment_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return next(
            row for row in reader if str(row.get("reference_label", "")).strip().lower() == reference_label.strip().lower()
        )


def _read_trigger_row(session_dir: Path, raw_label: str) -> dict[str, str]:
    trigger_path = session_dir / "trigger_segments.csv"
    with trigger_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return next(row for row in reader if str(row.get("raw_label", "")).strip().lower() == raw_label.strip().lower())


def _records_from_trigger_row(
    *,
    trigger_row: dict[str, str],
    arrays: dict[str, np.ndarray],
) -> list[TriggerFrameRecord]:
    frame_indices = json.loads(trigger_row["raw_frame_indices"])
    motion_curve = json.loads(trigger_row["motion_energy_curve"])
    records: list[TriggerFrameRecord] = []
    for offset, frame_index in enumerate(frame_indices):
        idx = int(frame_index)
        records.append(
            TriggerFrameRecord(
                frame_index=idx,
                frame_landmarks=FrameLandmarks(
                    left_hand=arrays["raw_left_hand"][idx].astype(np.float32),
                    right_hand=arrays["raw_right_hand"][idx].astype(np.float32),
                    pose=arrays["raw_pose"][idx].astype(np.float32),
                    mouth_center=arrays["raw_mouth_center"][idx].astype(np.float32),
                    chin=arrays["raw_chin"][idx].astype(np.float32),
                    left_hand_mask=arrays["left_hand_mask"][idx].astype(np.float32),
                    right_hand_mask=arrays["right_hand_mask"][idx].astype(np.float32),
                    pose_mask=arrays["pose_mask"][idx].astype(np.float32),
                    mouth_mask=arrays["mouth_mask"][idx].astype(np.float32),
                    chin_mask=arrays["chin_mask"][idx].astype(np.float32),
                ),
                motion_energy=float(motion_curve[offset]),
                left_hand_present=bool(np.max(arrays["left_hand_mask"][idx]) > 0),
                right_hand_present=bool(np.max(arrays["right_hand_mask"][idx]) > 0),
                pose_present=bool(np.max(arrays["pose_mask"][idx]) > 0),
            )
        )
    return records


def _configure_engine(
    *,
    checkpoint_path: Path,
    gesture_profile_path: Path,
    enable_mother_fix: bool,
    enable_you_i_fix: bool,
) -> MultibranchSequenceEngine:
    engine = MultibranchSequenceEngine(
        checkpoint_path=checkpoint_path,
        sequence_length=30,
        confidence_threshold=0.35,
        gesture_profile_path=gesture_profile_path,
        mode="trigger_based",
    )
    engine.pose_local_anchor = "torso_center"
    engine.enable_mother_nose_z_calibration = bool(enable_mother_fix)
    engine.enable_you_i_leftbody_calibration = bool(enable_you_i_fix)
    engine.enable_you_like_frontloaded_rescore = False
    engine.mother_you_left_face_prototypes = engine._load_mother_you_left_face_prototypes()
    engine.you_i_left_body_prototypes = engine._load_you_i_left_body_prototypes()
    return engine


def _score_stage(
    *,
    engine: MultibranchSequenceEngine,
    sequence: np.ndarray,
) -> dict[str, object]:
    probabilities, logit_scores, top_candidates, top_logits, raw_label, raw_confidence = engine._predict_sequence_scores(sequence)
    if len(top_candidates) >= 2:
        top_margin = float(top_candidates[0][1] - top_candidates[1][1])
    elif len(top_candidates) == 1:
        top_margin = float(top_candidates[0][1])
    else:
        top_margin = 0.0
    you_logit = float(logit_scores.get("you", 0.0))
    like_logit = float(logit_scores.get("like", 0.0))
    return {
        "raw_label": str(raw_label),
        "raw_confidence": _safe_float(raw_confidence),
        "top_margin": _safe_float(top_margin),
        "top_candidates": [
            {"label": label, "confidence": _safe_float(score)}
            for label, score in top_candidates[:5]
        ],
        "top_logits": [
            {"label": label, "logit": _safe_float(score)}
            for label, score in top_logits[:5]
        ],
        "you_like_logit_gap": _safe_float(you_logit - like_logit),
        "you_like_probability_gap": _safe_float(float(probabilities.get("you", 0.0)) - float(probabilities.get("like", 0.0))),
        "you_logit": _safe_float(you_logit),
        "like_logit": _safe_float(like_logit),
    }


def _apply_symmetric_you_like_delta(
    engine: MultibranchSequenceEngine,
    *,
    logit_scores: dict[str, float],
    delta: float,
) -> dict[str, object]:
    updated = dict(logit_scores)
    updated["you"] = float(updated.get("you", 0.0)) + (float(delta) / 2.0)
    updated["like"] = float(updated.get("like", 0.0)) - (float(delta) / 2.0)
    probabilities = engine._probabilities_from_logit_scores(updated)
    top_candidates = engine._top_candidates_from_probabilities(probabilities, limit=5)
    top_logits = engine._top_logits_from_scores(updated, limit=5)
    if len(top_candidates) >= 2:
        top_margin = float(top_candidates[0][1] - top_candidates[1][1])
    elif len(top_candidates) == 1:
        top_margin = float(top_candidates[0][1])
    else:
        top_margin = 0.0
    return {
        "delta": _safe_float(delta),
        "raw_label": str(top_candidates[0][0]) if top_candidates else "",
        "raw_confidence": _safe_float(top_candidates[0][1]) if top_candidates else 0.0,
        "top_margin": _safe_float(top_margin),
        "top_candidates": [
            {"label": label, "confidence": _safe_float(score)}
            for label, score in top_candidates[:5]
        ],
        "top_logits": [
            {"label": label, "logit": _safe_float(score)}
            for label, score in top_logits[:5]
        ],
        "you_like_logit_gap": _safe_float(float(updated.get("you", 0.0)) - float(updated.get("like", 0.0))),
        "you_like_probability_gap": _safe_float(float(probabilities.get("you", 0.0)) - float(probabilities.get("like", 0.0))),
    }


def _feature_group_slices(feature_spec: dict[str, object]) -> dict[str, slice]:
    landmarks = feature_spec["components"]["landmarks"]
    location = feature_spec["components"]["location"]
    motion = feature_spec["components"]["motion"]
    left_finger_start = int(landmarks["start"]) + int(feature_spec["landmark_graph_dim"])
    return {
        "left_finger_states": slice(left_finger_start, left_finger_start + 5),
        "left_face_vectors": slice(int(location["start"]), int(location["start"]) + 9),
        "left_body_vectors": slice(int(location["start"]) + 9, int(location["start"]) + 18),
        "location_all": slice(int(location["start"]), int(location["end"])),
        "skeleton_all": slice(int(landmarks["start"]), int(landmarks["end"])),
        "motion_all": slice(int(motion["start"]), int(motion["end"])),
    }


def _mean_vector(feature_vectors: np.ndarray, sampled_frame_indices: list[int], feature_slice: slice) -> np.ndarray:
    return feature_vectors[np.asarray(sampled_frame_indices, dtype=np.int32)][:, feature_slice].mean(axis=0)


def _rebuild_pose_local_sequence(
    *,
    feature_vectors: np.ndarray,
    normalized_pose: np.ndarray,
    sampled_frame_indices: list[int],
) -> np.ndarray:
    sampled = np.asarray(sampled_frame_indices, dtype=np.int32)
    sequence = feature_vectors[sampled].astype(np.float32, copy=True)
    channels = 3
    left_xyz = 21 * channels
    right_xyz = 21 * channels
    pose_coord_start = left_xyz + right_xyz
    pose_coord_end = pose_coord_start + (9 * channels)
    for row_index, frame_index in enumerate(sampled.tolist()):
        sequence[row_index, pose_coord_start:pose_coord_end] = normalized_pose[frame_index].reshape(-1).astype(np.float32)
    return sequence


def main() -> None:
    args = build_parser().parse_args()
    baseline_session = Path(args.baseline_session).resolve()
    current_session = Path(args.current_session).resolve()
    exact_classification_path = Path(args.exact_classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    gesture_profile_path = Path(args.gesture_profile).resolve()
    output_path = Path(args.output_json).resolve()

    baseline_alignment_you = _read_alignment_row(baseline_session, "you")
    current_alignment_you = _read_alignment_row(current_session, "you")
    current_trigger_you = _read_trigger_row(current_session, "you")
    exact_payload = _load_json(exact_classification_path)
    exact_you_row = next(
        row
        for row in exact_payload["classifications"]
        if int(row.get("mirror_input", -1)) == 1 and str(row.get("token", "")).strip().lower() == "you"
    )
    exact_like_row = next(
        row
        for row in exact_payload["classifications"]
        if int(row.get("mirror_input", -1)) == 1 and str(row.get("token", "")).strip().lower() == "like"
    )

    cache_path = cache_dir / "continuous_feature_cache_mirror1.npz"
    _meta, arrays = load_continuous_feature_cache(cache_path)
    feature_vectors = arrays["feature_vectors"].astype(np.float32)
    serving_records = _records_from_trigger_row(trigger_row=current_trigger_you, arrays=arrays)

    base_engine = _configure_engine(
        checkpoint_path=checkpoint_path,
        gesture_profile_path=gesture_profile_path,
        enable_mother_fix=False,
        enable_you_i_fix=False,
    )
    mother_engine = _configure_engine(
        checkpoint_path=checkpoint_path,
        gesture_profile_path=gesture_profile_path,
        enable_mother_fix=True,
        enable_you_i_fix=False,
    )
    current_engine = _configure_engine(
        checkpoint_path=checkpoint_path,
        gesture_profile_path=gesture_profile_path,
        enable_mother_fix=True,
        enable_you_i_fix=True,
    )

    serving_sequence, serving_build_debug = current_engine._build_trigger_sequence(serving_records)
    base_stage = _score_stage(engine=base_engine, sequence=serving_sequence)
    mother_stage = _score_stage(engine=mother_engine, sequence=serving_sequence)
    you_i_stage = _score_stage(engine=current_engine, sequence=serving_sequence)
    exact_you_sequence = _rebuild_pose_local_sequence(
        feature_vectors=feature_vectors,
        normalized_pose=arrays["normalized_pose"],
        sampled_frame_indices=exact_you_row["sampled_frame_indices"],
    )
    exact_you_stage = _score_stage(engine=current_engine, sequence=exact_you_sequence)

    current_logit_scores = {
        item["label"]: float(item["logit"])
        for item in you_i_stage["top_logits"]
    }
    _, full_logit_scores, _, _, _, _ = current_engine._predict_sequence_scores(serving_sequence)
    sweep_rows: list[dict[str, object]] = []
    recommended_row = None
    for delta in np.arange(0.10, 1.21, 0.02):
        candidate = _apply_symmetric_you_like_delta(
            current_engine,
            logit_scores=full_logit_scores,
            delta=float(delta),
        )
        sweep_rows.append(candidate)
        if (
            recommended_row is None
            and str(candidate["raw_label"]).strip().lower() == "you"
            and float(candidate["raw_confidence"]) > current_engine.trigger_emit_confidence_threshold
            and float(candidate["top_margin"]) > current_engine.trigger_min_top_margin
        ):
            recommended_row = candidate

    feature_slices = _feature_group_slices(current_engine.feature_spec)
    serving_sampled_indices = json.loads(current_alignment_you["sampled_frame_indices"])
    feature_bridge: dict[str, dict[str, float]] = {}
    exact_you_aligned = _rebuild_pose_local_sequence(
        feature_vectors=feature_vectors,
        normalized_pose=arrays["normalized_pose"],
        sampled_frame_indices=exact_you_row["sampled_frame_indices"],
    )
    exact_like_aligned = _rebuild_pose_local_sequence(
        feature_vectors=feature_vectors,
        normalized_pose=arrays["normalized_pose"],
        sampled_frame_indices=exact_like_row["sampled_frame_indices"],
    )
    for name, feature_slice in feature_slices.items():
        serving_mean = serving_sequence[:, feature_slice].mean(axis=0)
        exact_you_mean = exact_you_aligned[:, feature_slice].mean(axis=0)
        exact_like_mean = exact_like_aligned[:, feature_slice].mean(axis=0)
        distance_to_exact_you = float(np.linalg.norm(serving_mean - exact_you_mean))
        distance_to_exact_like = float(np.linalg.norm(serving_mean - exact_like_mean))
        feature_bridge[name] = {
            "serving_to_exact_you_l2": _safe_float(distance_to_exact_you),
            "serving_to_exact_like_l2": _safe_float(distance_to_exact_like),
            "exact_you_advantage_l2": _safe_float(distance_to_exact_like - distance_to_exact_you),
        }

    stage_report = {
        "baseline_artifact_session": str(baseline_session),
        "current_artifact_session": str(current_session),
        "baseline_trigger_artifact": {
            "raw_label": str(baseline_alignment_you["raw_label"]),
            "raw_confidence": _safe_float(float(baseline_alignment_you["raw_confidence"])),
            "top_margin": _safe_float(float(baseline_alignment_you["top_margin"])),
            "top_logits": json.loads(baseline_alignment_you["top_logits"]),
            "top_candidates": json.loads(baseline_alignment_you["top_candidates"]),
        },
        "current_trigger_artifact": {
            "raw_label": str(current_alignment_you["raw_label"]),
            "raw_confidence": _safe_float(float(current_alignment_you["raw_confidence"])),
            "top_margin": _safe_float(float(current_alignment_you["top_margin"])),
            "top_logits": json.loads(current_alignment_you["top_logits"]),
            "top_candidates": json.loads(current_alignment_you["top_candidates"]),
        },
        "serving_stage_scores": {
            "raw_model_no_pairwise_fix": base_stage,
            "after_mother_fix_only": mother_stage,
            "after_mother_and_you_i_fix": you_i_stage,
            "exact_you_reference_after_you_i_fix": exact_you_stage,
        },
        "stage_deltas": {
            "you_logit_base_to_you_i": _safe_float(float(you_i_stage["you_logit"]) - float(base_stage["you_logit"])),
            "like_logit_base_to_you_i": _safe_float(float(you_i_stage["like_logit"]) - float(base_stage["like_logit"])),
            "you_like_logit_gap_base_to_you_i": _safe_float(
                float(you_i_stage["you_like_logit_gap"]) - float(base_stage["you_like_logit_gap"])
            ),
            "you_like_logit_gap_exact_reference_minus_serving": _safe_float(
                float(exact_you_stage["you_like_logit_gap"]) - float(you_i_stage["you_like_logit_gap"])
            ),
            "like_logit_serving_minus_exact_reference": _safe_float(
                float(you_i_stage["like_logit"]) - float(exact_you_stage["like_logit"])
            ),
        },
        "feature_bridge_to_exact_spans": feature_bridge,
        "you_i_leftbody_gate": {
            "baseline_gap": _safe_float(current_engine._you_i_leftbody_gap(serving_sequence)[0]),
            "adjusted_gap": _safe_float(current_engine._you_i_leftbody_gap(serving_sequence)[1]),
        },
        "recommended_pairwise_fix": recommended_row,
        "pairwise_delta_sweep": sweep_rows,
        "root_cause_hypothesis": (
            "The serving you segment is already much closer to exact-span you than exact-span like across every coarse scorer group, "
            "so the remaining defect is not a broad feature-window mismatch. "
            "After the accepted you/i fix, the dominant residual is an under-separated you-vs-like pair: "
            "you stays top1, but like retains a logit that is materially higher than the exact-you reference, "
            "so the top2 spacing never crosses the serving gate. "
            "The smallest implementable repair is therefore a gated symmetric you/like pairwise spacing correction."
        ),
    }
    output_path.write_text(json.dumps(stage_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
