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
    _records_from_trigger_row,
    _safe_float,
    _score_stage,
)
from src.dataset.continuous_feature_cache import load_continuous_feature_cache  # noqa: E402


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Analyze want trigger segment counterfactuals under anchor and pose-coordinate swaps.")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gesture-profile", required=True)
    parser.add_argument("--segment-id", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _load_trigger_row(session_dir: Path, segment_id: int) -> dict[str, str]:
    trigger_path = session_dir / "trigger_segments.csv"
    with trigger_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(row for row in csv.DictReader(handle) if int(row.get("segment_id", -1)) == int(segment_id))


def _read_alignment_row(session_dir: Path, segment_id: int) -> dict[str, str]:
    alignment_path = session_dir / "trigger_segment_alignment.csv"
    with alignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(row for row in csv.DictReader(handle) if int(row.get("segment_id", -1)) == int(segment_id))


def _pose_coord_slice() -> slice:
    pose_coord_start = (21 * 3) + (21 * 3)
    pose_coord_end = pose_coord_start + (9 * 3)
    return slice(pose_coord_start, pose_coord_end)


def _stage_metrics(stage: dict[str, object]) -> dict[str, object]:
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


def _build_engine(checkpoint_path: Path, gesture_profile_path: Path):
    engine = _configure_engine(
        checkpoint_path=checkpoint_path,
        gesture_profile_path=gesture_profile_path,
        enable_mother_fix=True,
        enable_you_i_fix=True,
    )
    engine.enable_you_like_pairwise_calibration = True
    engine.you_like_pairwise_delta = 0.98
    engine.enable_like_i_pairwise_calibration = True
    engine.like_i_pairwise_delta = 2.78
    return engine


def main() -> None:
    args = build_parser().parse_args()
    session_dir = Path(args.session_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    gesture_profile_path = Path(args.gesture_profile).resolve()
    output_path = Path(args.output_json).resolve()

    trigger_row = _load_trigger_row(session_dir, args.segment_id)
    alignment_row = _read_alignment_row(session_dir, args.segment_id)
    cache_path = cache_dir / "continuous_feature_cache_mirror1.npz"
    _metadata, arrays = load_continuous_feature_cache(cache_path)
    records = _records_from_trigger_row(trigger_row=trigger_row, arrays=arrays)

    torso_engine = _build_engine(checkpoint_path, gesture_profile_path)
    torso_engine.pose_local_anchor = "torso_center"
    mid_engine = _build_engine(checkpoint_path, gesture_profile_path)
    mid_engine.pose_local_anchor = "mid_shoulder"

    torso_sequence, torso_debug = torso_engine._build_trigger_sequence(records)
    mid_sequence, mid_debug = mid_engine._build_trigger_sequence(records)
    pose_slice = _pose_coord_slice()
    torso_with_mid_pose = torso_sequence.astype(np.float32, copy=True)
    torso_with_mid_pose[:, pose_slice] = mid_sequence[:, pose_slice]

    output_payload = {
        "session_dir": str(session_dir),
        "segment_id": int(args.segment_id),
        "reference_label": str(alignment_row.get("reference_label", "")),
        "decision_status": str(alignment_row.get("decision_status", "")),
        "current_raw_label": str(alignment_row.get("raw_label", "")),
        "results": {
            "torso_center": _stage_metrics(_score_stage(engine=torso_engine, sequence=torso_sequence)),
            "mid_shoulder": _stage_metrics(_score_stage(engine=mid_engine, sequence=mid_sequence)),
            "torso_with_mid_pose_coords_only": _stage_metrics(_score_stage(engine=torso_engine, sequence=torso_with_mid_pose)),
        },
        "sampled_frame_indices": {
            "torso_center": torso_debug.get("sampled_frame_indices", []),
            "mid_shoulder": mid_debug.get("sampled_frame_indices", []),
        },
        "conclusion": (
            "If torso_with_mid_pose_coords_only materially improves want on the live segment, the smallest serving-side "
            "counterfactual is again the pose-local pose reconstruction. If it stays no_sign or student with a large gap, "
            "the live branch is still deeper than the exact-span pose-coordinate defect."
        ),
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
