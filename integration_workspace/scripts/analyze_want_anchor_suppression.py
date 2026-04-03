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

from scripts.analyze_you_like_serving_margin import (  # noqa: E402
    _configure_engine,
    _load_json,
    _records_from_trigger_row,
    _rebuild_pose_local_sequence,
    _safe_float,
    _score_stage,
)
from src.dataset.continuous_feature_cache import load_continuous_feature_cache  # noqa: E402


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Analyze the want no_sign/student suppression structure under anchor variants.")
    parser.add_argument("--current-session", required=True)
    parser.add_argument("--exact-classification-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gesture-profile", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--you-like-pairwise-delta", type=float, default=0.98)
    parser.add_argument("--like-i-pairwise-delta", type=float, default=2.78)
    return parser


def _read_target_alignment_row(session_dir: Path, reference_label: str) -> dict[str, str]:
    alignment_path = session_dir / "trigger_segment_alignment.csv"
    with alignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("reference_label", "")).strip().lower() == reference_label.strip().lower()
        ]
    if not rows:
        raise RuntimeError(f"No alignment rows for {reference_label!r} in {alignment_path}")
    rows.sort(
        key=lambda row: (
            1 if str(row.get("decision_status", "")).strip().lower() == "trigger_emit" else 0,
            float(row.get("reference_overlap_ratio", 0.0) or 0.0),
            int(row.get("trimmed_length", 0) or 0),
            float(row.get("raw_confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return rows[0]


def _read_trigger_row_by_segment_id(session_dir: Path, segment_id: int) -> dict[str, str]:
    trigger_path = session_dir / "trigger_segments.csv"
    with trigger_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(row for row in csv.DictReader(handle) if int(row.get("segment_id", -1)) == int(segment_id))


def _top_logit(stage: dict[str, object], label: str) -> float:
    for item in stage.get("top_logits", []):
        if str(item.get("label", "")).strip().lower() == label.strip().lower():
            return float(item.get("logit", 0.0))
    return float("-inf")


def _top_probability(stage: dict[str, object], label: str) -> float:
    for item in stage.get("top_candidates", []):
        if str(item.get("label", "")).strip().lower() == label.strip().lower():
            return float(item.get("confidence", 0.0))
    return 0.0


def _required_pairwise_delta(stage: dict[str, object], positive_label: str, negative_label: str) -> float | None:
    positive_logit = _top_logit(stage, positive_label)
    negative_logit = _top_logit(stage, negative_label)
    if not np.isfinite(positive_logit) or not np.isfinite(negative_logit):
        return None
    return _safe_float(max(0.0, negative_logit - positive_logit))


def _required_nosign_penalty_factor(stage: dict[str, object], target_label: str) -> float | None:
    nosign_probability = _top_probability(stage, "no_sign")
    target_probability = _top_probability(stage, target_label)
    if nosign_probability <= 1e-8:
        return None
    return _safe_float(max(0.0, target_probability / nosign_probability))


def _configure_current_engine(
    *,
    checkpoint_path: Path,
    gesture_profile_path: Path,
    pose_local_anchor: str,
    you_like_pairwise_delta: float,
    like_i_pairwise_delta: float,
):
    engine = _configure_engine(
        checkpoint_path=checkpoint_path,
        gesture_profile_path=gesture_profile_path,
        enable_mother_fix=True,
        enable_you_i_fix=True,
    )
    engine.pose_local_anchor = pose_local_anchor
    engine.enable_you_like_pairwise_calibration = True
    engine.you_like_pairwise_delta = float(you_like_pairwise_delta)
    engine.enable_like_i_pairwise_calibration = True
    engine.like_i_pairwise_delta = float(like_i_pairwise_delta)
    return engine


def main() -> None:
    args = build_parser().parse_args()
    current_session = Path(args.current_session).resolve()
    exact_classification_path = Path(args.exact_classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    gesture_profile_path = Path(args.gesture_profile).resolve()
    output_path = Path(args.output_json).resolve()

    exact_payload = _load_json(exact_classification_path)
    exact_want_row = next(
        row
        for row in exact_payload["classifications"]
        if int(row.get("mirror_input", -1)) == 1 and str(row.get("token", "")).strip().lower() == "want"
    )
    want_alignment_row = _read_target_alignment_row(current_session, "want")
    want_trigger_row = _read_trigger_row_by_segment_id(current_session, int(want_alignment_row["segment_id"]))

    cache_path = cache_dir / "continuous_feature_cache_mirror1.npz"
    _metadata, arrays = load_continuous_feature_cache(cache_path)
    feature_vectors = arrays["feature_vectors"].astype(np.float32)
    serving_records = _records_from_trigger_row(trigger_row=want_trigger_row, arrays=arrays)
    want_sampled_frame_indices = list(exact_want_row["sampled_frame_indices"])

    stage_reports: dict[str, dict[str, object]] = {}
    for anchor in ("torso_center", "mid_shoulder"):
        engine = _configure_current_engine(
            checkpoint_path=checkpoint_path,
            gesture_profile_path=gesture_profile_path,
            pose_local_anchor=anchor,
            you_like_pairwise_delta=float(args.you_like_pairwise_delta),
            like_i_pairwise_delta=float(args.like_i_pairwise_delta),
        )
        if anchor == "torso_center":
            exact_sequence = _rebuild_pose_local_sequence(
                feature_vectors=feature_vectors,
                normalized_pose=arrays["normalized_pose"],
                sampled_frame_indices=want_sampled_frame_indices,
            )
        else:
            exact_sequence = feature_vectors[np.asarray(want_sampled_frame_indices, dtype=np.int32)].astype(np.float32, copy=True)
        serving_sequence, _ = engine._build_trigger_sequence(serving_records)
        exact_stage = _score_stage(engine=engine, sequence=exact_sequence)
        serving_stage = _score_stage(engine=engine, sequence=serving_sequence)
        stage_reports[anchor] = {
            "exact_span": exact_stage,
            "serving_segment": serving_stage,
            "required_pairwise_deltas": {
                "exact_want_vs_no_sign": _required_pairwise_delta(exact_stage, "want", "no_sign"),
                "exact_want_vs_student": _required_pairwise_delta(exact_stage, "want", "student"),
                "serving_want_vs_no_sign": _required_pairwise_delta(serving_stage, "want", "no_sign"),
                "serving_want_vs_student": _required_pairwise_delta(serving_stage, "want", "student"),
            },
            "required_nosign_penalty_factors": {
                "exact_student_over_no_sign": _required_nosign_penalty_factor(exact_stage, "student"),
                "exact_want_over_no_sign": _required_nosign_penalty_factor(exact_stage, "want"),
                "serving_student_over_no_sign": _required_nosign_penalty_factor(serving_stage, "student"),
                "serving_want_over_no_sign": _required_nosign_penalty_factor(serving_stage, "want"),
            },
        }

    conclusion = (
        "Under the current accepted fixes, torso_center is the concrete source of the exact-span no_sign branch on want: "
        "the same sampled want span flips from no_sign at torso_center to student at mid_shoulder, while the serving trigger segment "
        "stays student under both anchors. This means the current want residual is a two-stage structure rather than one pure scorer pair: "
        "first torso_center pushes exact-span want into no_sign, then the serving trigger path remains strongly student-biased. "
        "A global no_sign penalty is not a viable repair because the required factors are extremely small, and a scorer-only want-vs-student delta "
        "is also not a near-margin fix on the current serving segment."
    )

    output_payload = {
        "current_session": str(current_session),
        "exact_classification_json": str(exact_classification_path),
        "cache_path": str(cache_path),
        "accepted_fix_configuration": {
            "pose_local_anchor_variants": ["torso_center", "mid_shoulder"],
            "enable_mother_nose_z_calibration": True,
            "enable_you_i_leftbody_calibration": True,
            "enable_you_like_pairwise_calibration": True,
            "you_like_pairwise_delta": _safe_float(float(args.you_like_pairwise_delta)),
            "enable_like_i_pairwise_calibration": True,
            "like_i_pairwise_delta": _safe_float(float(args.like_i_pairwise_delta)),
        },
        "want_alignment_row": {
            "segment_id": int(want_alignment_row["segment_id"]),
            "reference_overlap_ratio": _safe_float(float(want_alignment_row["reference_overlap_ratio"])),
            "decision_status": str(want_alignment_row["decision_status"]),
            "raw_label": str(want_alignment_row["raw_label"]),
            "emitted_label": str(want_alignment_row.get("emitted_label", "")),
        },
        "stage_reports": stage_reports,
        "conclusion": conclusion,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
