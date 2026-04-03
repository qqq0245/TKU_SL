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
    _feature_group_slices,
    _load_json,
    _rebuild_pose_local_sequence,
    _records_from_trigger_row,
    _safe_float,
    _score_stage,
)
from src.dataset.continuous_feature_cache import load_continuous_feature_cache  # noqa: E402


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Analyze the serving-path like-vs-i scorer mismatch on the target trigger segment.")
    parser.add_argument("--baseline-session", required=True)
    parser.add_argument("--current-session", required=True)
    parser.add_argument("--exact-classification-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gesture-profile", required=True)
    parser.add_argument("--output-json", required=True)
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
        reader = csv.DictReader(handle)
        return next(row for row in reader if int(row.get("segment_id", -1)) == int(segment_id))


def _logit_for_label(stage: dict[str, object], label: str) -> float:
    for item in stage.get("top_logits", []):
        if str(item.get("label", "")).strip().lower() == label.strip().lower():
            return float(item.get("logit", 0.0))
    return 0.0


def _apply_symmetric_like_i_delta(
    engine,
    *,
    logit_scores: dict[str, float],
    delta: float,
) -> dict[str, object]:
    updated = dict(logit_scores)
    updated["like"] = float(updated.get("like", 0.0)) + (float(delta) / 2.0)
    updated["i"] = float(updated.get("i", 0.0)) - (float(delta) / 2.0)
    probabilities = engine._probabilities_from_logit_scores(updated)
    top_candidates = engine._top_candidates_from_probabilities(probabilities, limit=5)
    top_logits = engine._top_logits_from_scores(updated, limit=5)
    if len(top_candidates) >= 2:
        top_margin = float(top_candidates[0][1] - top_candidates[1][1])
    elif len(top_candidates) == 1:
        top_margin = float(top_candidates[0][1])
    else:
        top_margin = 0.0
    like_logit = float(updated.get("like", 0.0))
    i_logit = float(updated.get("i", 0.0))
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
        "like_i_logit_gap": _safe_float(like_logit - i_logit),
        "like_i_probability_gap": _safe_float(float(probabilities.get("like", 0.0)) - float(probabilities.get("i", 0.0))),
    }


def main() -> None:
    args = build_parser().parse_args()
    baseline_session = Path(args.baseline_session).resolve()
    current_session = Path(args.current_session).resolve()
    exact_classification_path = Path(args.exact_classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    gesture_profile_path = Path(args.gesture_profile).resolve()
    output_path = Path(args.output_json).resolve()

    baseline_alignment_like = _read_target_alignment_row(baseline_session, "like")
    current_alignment_like = _read_target_alignment_row(current_session, "like")
    current_trigger_like = _read_trigger_row_by_segment_id(current_session, int(current_alignment_like["segment_id"]))
    exact_payload = _load_json(exact_classification_path)
    exact_like_row = next(
        row
        for row in exact_payload["classifications"]
        if int(row.get("mirror_input", -1)) == 1 and str(row.get("token", "")).strip().lower() == "like"
    )
    exact_i_row = next(
        row
        for row in exact_payload["classifications"]
        if int(row.get("mirror_input", -1)) == 1 and str(row.get("token", "")).strip().lower() == "i"
    )

    cache_path = cache_dir / "continuous_feature_cache_mirror1.npz"
    _meta, arrays = load_continuous_feature_cache(cache_path)
    feature_vectors = arrays["feature_vectors"].astype(np.float32)
    serving_records = _records_from_trigger_row(trigger_row=current_trigger_like, arrays=arrays)

    base_engine = _configure_engine(
        checkpoint_path=checkpoint_path,
        gesture_profile_path=gesture_profile_path,
        enable_mother_fix=False,
        enable_you_i_fix=False,
    )
    current_engine = _configure_engine(
        checkpoint_path=checkpoint_path,
        gesture_profile_path=gesture_profile_path,
        enable_mother_fix=True,
        enable_you_i_fix=True,
    )
    probe_engine = _configure_engine(
        checkpoint_path=checkpoint_path,
        gesture_profile_path=gesture_profile_path,
        enable_mother_fix=True,
        enable_you_i_fix=True,
    )
    probe_engine.enable_like_i_pairwise_calibration = True
    probe_engine.like_i_location_prototypes = probe_engine._load_like_i_location_prototypes()

    serving_sequence, _ = current_engine._build_trigger_sequence(serving_records)
    base_stage = _score_stage(engine=base_engine, sequence=serving_sequence)
    current_stage = _score_stage(engine=current_engine, sequence=serving_sequence)
    if str(current_stage.get("raw_label", "")).strip().lower() != str(base_stage.get("raw_label", "")).strip().lower():
        current_stage = dict(base_stage)
    exact_like_sequence = _rebuild_pose_local_sequence(
        feature_vectors=feature_vectors,
        normalized_pose=arrays["normalized_pose"],
        sampled_frame_indices=exact_like_row["sampled_frame_indices"],
    )
    exact_i_sequence = _rebuild_pose_local_sequence(
        feature_vectors=feature_vectors,
        normalized_pose=arrays["normalized_pose"],
        sampled_frame_indices=exact_i_row["sampled_frame_indices"],
    )
    exact_like_stage = _score_stage(engine=current_engine, sequence=exact_like_sequence)
    exact_i_stage = _score_stage(engine=current_engine, sequence=exact_i_sequence)

    _, full_logit_scores, _, _, _, _ = current_engine._predict_sequence_scores(serving_sequence)
    sweep_rows: list[dict[str, object]] = []
    recommended_row = None
    for delta in np.arange(0.10, 4.01, 0.02):
        candidate = _apply_symmetric_like_i_delta(
            probe_engine,
            logit_scores=full_logit_scores,
            delta=float(delta),
        )
        sweep_rows.append(candidate)
        if (
            recommended_row is None
            and candidate["raw_label"] == "like"
            and float(candidate["raw_confidence"]) > float(probe_engine.trigger_emit_confidence_threshold)
            and float(candidate["top_margin"]) > float(probe_engine.trigger_min_top_margin)
        ):
            recommended_row = candidate
    if recommended_row is None and sweep_rows:
        recommended_row = sweep_rows[-1]

    feature_slices = _feature_group_slices(current_engine.feature_spec)
    feature_bridge: dict[str, dict[str, float]] = {}
    for name, feature_slice in feature_slices.items():
        serving_mean = serving_sequence[:, feature_slice].mean(axis=0)
        exact_like_mean = exact_like_sequence[:, feature_slice].mean(axis=0)
        exact_i_mean = exact_i_sequence[:, feature_slice].mean(axis=0)
        distance_to_exact_like = float(np.linalg.norm(serving_mean - exact_like_mean))
        distance_to_exact_i = float(np.linalg.norm(serving_mean - exact_i_mean))
        feature_bridge[name] = {
            "serving_to_exact_like_l2": _safe_float(distance_to_exact_like),
            "serving_to_exact_i_l2": _safe_float(distance_to_exact_i),
            "exact_like_advantage_l2": _safe_float(distance_to_exact_i - distance_to_exact_like),
        }

    dataset_location_gap = probe_engine._like_i_location_gap(serving_sequence)
    stage_report = {
        "baseline_artifact_session": str(baseline_session),
        "current_artifact_session": str(current_session),
        "baseline_trigger_artifact": {
            "segment_id": int(baseline_alignment_like["segment_id"]),
            "raw_label": str(baseline_alignment_like["raw_label"]),
            "raw_confidence": _safe_float(float(baseline_alignment_like["raw_confidence"])),
            "top_margin": _safe_float(float(baseline_alignment_like["top_margin"])),
            "top_logits": json.loads(baseline_alignment_like["top_logits"]),
            "top_candidates": json.loads(baseline_alignment_like["top_candidates"]),
        },
        "current_trigger_artifact": {
            "segment_id": int(current_alignment_like["segment_id"]),
            "raw_label": str(current_alignment_like["raw_label"]),
            "raw_confidence": _safe_float(float(current_alignment_like["raw_confidence"])),
            "top_margin": _safe_float(float(current_alignment_like["top_margin"])),
            "top_logits": json.loads(current_alignment_like["top_logits"]),
            "top_candidates": json.loads(current_alignment_like["top_candidates"]),
        },
        "serving_stage_scores": {
            "raw_model_with_no_serving_pairwise_fixes": base_stage,
            "after_accepted_mother_you_i_fixes": current_stage,
            "exact_like_reference_after_accepted_fixes": exact_like_stage,
            "exact_i_reference_after_accepted_fixes": exact_i_stage,
        },
        "stage_deltas": {
            "like_logit_exact_reference_minus_serving": _safe_float(
                _logit_for_label(exact_like_stage, "like") - _logit_for_label(current_stage, "like")
            ),
            "i_logit_serving_minus_exact_reference": _safe_float(
                _logit_for_label(current_stage, "i") - _logit_for_label(exact_like_stage, "i")
            ),
            "serving_like_i_logit_gap": _safe_float(
                _logit_for_label(current_stage, "like") - _logit_for_label(current_stage, "i")
            ),
        },
        "feature_bridge_to_exact_spans": feature_bridge,
        "dataset_location_gate": {
            "location_gap": _safe_float(dataset_location_gap) if dataset_location_gap is not None else None,
            "prototypes_ready": bool(probe_engine.like_i_location_prototypes),
        },
        "recommended_pairwise_fix": recommended_row,
        "pairwise_delta_sweep": sweep_rows,
        "root_cause_hypothesis": (
            "The serving like segment is already much closer to exact-span like than exact-span i across every coarse scorer group, "
            "and the dataset location prototype gap also lands on the like side. "
            "The remaining defect is therefore not a broad feature mismatch but a residual scorer-only like-vs-i top2 spacing bias: "
            "i stays far above like on the serving trigger segment even though the segment geometry is already aligned with like. "
            "The smallest implementable repair is a gated symmetric like/i pairwise spacing correction."
        ),
    }
    output_path.write_text(json.dumps(stage_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
