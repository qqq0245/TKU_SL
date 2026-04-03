from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_you_like_serving_margin import (  # noqa: E402
    _configure_engine,
    _load_json,
    _rebuild_pose_local_sequence,
    _safe_float,
    _score_stage,
)
from src.dataset.continuous_feature_cache import load_continuous_feature_cache  # noqa: E402


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Analyze which exact-span feature groups drive want suppression into no_sign.")
    parser.add_argument("--exact-classification-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gesture-profile", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _want_stage_metrics(stage: dict[str, object]) -> dict[str, object]:
    top_logits = {str(item["label"]).strip().lower(): float(item["logit"]) for item in stage.get("top_logits", [])}
    top_candidates = {str(item["label"]).strip().lower(): float(item["confidence"]) for item in stage.get("top_candidates", [])}
    return {
        "raw_label": str(stage.get("raw_label", "")),
        "raw_confidence": _safe_float(float(stage.get("raw_confidence", 0.0))),
        "top_margin": _safe_float(float(stage.get("top_margin", 0.0))),
        "want_probability": _safe_float(top_candidates.get("want", 0.0)),
        "student_probability": _safe_float(top_candidates.get("student", 0.0)),
        "nosign_probability": _safe_float(top_candidates.get("no_sign", 0.0)),
        "want_logit": _safe_float(top_logits.get("want", float("-inf"))),
        "student_logit": _safe_float(top_logits.get("student", float("-inf"))),
        "nosign_logit": _safe_float(top_logits.get("no_sign", float("-inf"))),
        "want_minus_student_logit_gap": _safe_float(top_logits.get("want", float("-inf")) - top_logits.get("student", float("-inf"))),
        "want_minus_nosign_logit_gap": _safe_float(top_logits.get("want", float("-inf")) - top_logits.get("no_sign", float("-inf"))),
        "top_candidates": stage.get("top_candidates", []),
        "top_logits": stage.get("top_logits", []),
    }


def _sequence_copy(sequence: np.ndarray) -> np.ndarray:
    return sequence.astype(np.float32, copy=True)


def _slice_groups(feature_spec: dict[str, object]) -> dict[str, slice]:
    landmarks = feature_spec["components"]["landmarks"]
    location = feature_spec["components"]["location"]
    motion = feature_spec["components"]["motion"]
    pose_coord_start = (21 * 3) + (21 * 3)
    pose_coord_end = pose_coord_start + (9 * 3)
    return {
        "pose_coords_only": slice(pose_coord_start, pose_coord_end),
        "skeleton_all": slice(int(landmarks["start"]), int(landmarks["end"])),
        "location_all": slice(int(location["start"]), int(location["end"])),
        "motion_all": slice(int(motion["start"]), int(motion["end"])),
    }


def _score_sequence(engine, sequence: np.ndarray) -> dict[str, object]:
    return _want_stage_metrics(_score_stage(engine=engine, sequence=sequence))


def main() -> None:
    args = build_parser().parse_args()
    exact_classification_path = Path(args.exact_classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    gesture_profile_path = Path(args.gesture_profile).resolve()
    output_path = Path(args.output_json).resolve()

    exact_payload = _load_json(exact_classification_path)
    want_row = next(
        row
        for row in exact_payload["classifications"]
        if int(row.get("mirror_input", -1)) == 1 and str(row.get("token", "")).strip().lower() == "want"
    )
    sampled_frame_indices = list(want_row["sampled_frame_indices"])

    cache_path = cache_dir / "continuous_feature_cache_mirror1.npz"
    _metadata, arrays = load_continuous_feature_cache(cache_path)
    feature_vectors = arrays["feature_vectors"].astype(np.float32)

    engine = _configure_engine(
        checkpoint_path=checkpoint_path,
        gesture_profile_path=gesture_profile_path,
        enable_mother_fix=True,
        enable_you_i_fix=True,
    )
    engine.pose_local_anchor = "torso_center"
    engine.enable_you_like_pairwise_calibration = True
    engine.you_like_pairwise_delta = 0.98
    engine.enable_like_i_pairwise_calibration = True
    engine.like_i_pairwise_delta = 2.78

    mid_sequence = feature_vectors[np.asarray(sampled_frame_indices, dtype=np.int32)].astype(np.float32, copy=True)
    torso_sequence = _rebuild_pose_local_sequence(
        feature_vectors=feature_vectors,
        normalized_pose=arrays["normalized_pose"],
        sampled_frame_indices=sampled_frame_indices,
    )

    groups = _slice_groups(engine.feature_spec)
    results: dict[str, dict[str, object]] = {
        "mid_shoulder_base": _score_sequence(engine, mid_sequence),
        "torso_center_base": _score_sequence(engine, torso_sequence),
    }

    dominant_group = None
    for group_name, feature_slice in groups.items():
        torso_with_mid = _sequence_copy(torso_sequence)
        torso_with_mid[:, feature_slice] = mid_sequence[:, feature_slice]
        torso_key = f"torso_with_mid_{group_name}"
        results[torso_key] = _score_sequence(engine, torso_with_mid)

        mid_with_torso = _sequence_copy(mid_sequence)
        mid_with_torso[:, feature_slice] = torso_sequence[:, feature_slice]
        mid_key = f"mid_with_torso_{group_name}"
        results[mid_key] = _score_sequence(engine, mid_with_torso)

        if dominant_group is None:
            torso_raw = str(results["torso_center_base"]["raw_label"]).strip().lower()
            mid_raw = str(results["mid_shoulder_base"]["raw_label"]).strip().lower()
            torso_swapped_raw = str(results[torso_key]["raw_label"]).strip().lower()
            mid_swapped_raw = str(results[mid_key]["raw_label"]).strip().lower()
            if torso_raw == "no_sign" and torso_swapped_raw != "no_sign" and mid_raw == "student" and mid_swapped_raw == "no_sign":
                dominant_group = group_name

    output_payload = {
        "exact_classification_json": str(exact_classification_path),
        "cache_path": str(cache_path),
        "sampled_frame_indices": sampled_frame_indices,
        "results": results,
        "smallest_counterfactual_group": dominant_group,
        "conclusion": (
            "This exact-span counterfactual tests which feature group is sufficient to transfer the want suppression "
            "between mid_shoulder and torso_center. If swapping only pose_coords_only reproduces the label flip in both "
            "directions, the smallest actionable counterfactual is the pose-local pose reconstruction rather than a "
            "full location or motion rewrite."
        ),
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
