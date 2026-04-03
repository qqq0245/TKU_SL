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
    parser = ArgumentParser(description="Sweep live want serving windows and anchors on the trigger segment.")
    parser.add_argument("--session-dir", required=True)
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
        return next(row for row in csv.DictReader(handle) if int(row.get("segment_id", -1)) == int(segment_id))


def _sample_variants(frame_count: int) -> list[tuple[str, list[int]]]:
    def uniform_range(start: int, end: int, count: int = 30) -> list[int]:
        if end <= start:
            return [start for _ in range(count)]
        return [int(index) for index in np.linspace(start, end, count).round().astype(int).tolist()]

    last = max(frame_count - 1, 0)
    head_end = max(int(round(last * 0.74)), 0)
    tail_start = min(int(round(last * 0.26)), last)
    middle_start = min(int(round(last * 0.13)), last)
    middle_end = max(min(int(round(last * 0.87)), last), middle_start)
    center = last // 2
    center_half = 15
    middle_window_start = max(center - center_half, 0)
    middle_window_end = min(middle_window_start + 29, last)
    middle_window_start = max(middle_window_end - 29, 0)
    return [
        ("uniform_full", uniform_range(0, last)),
        ("uniform_head_75", uniform_range(0, head_end)),
        ("uniform_tail_75", uniform_range(tail_start, last)),
        ("uniform_middle_75", uniform_range(middle_start, middle_end)),
        ("first_30", uniform_range(0, min(29, last))),
        ("middle_30", uniform_range(middle_window_start, middle_window_end)),
        ("last_30", uniform_range(max(last - 29, 0), last)),
    ]


def _label_score(stage: dict[str, object], label: str) -> tuple[float, float]:
    probability = 0.0
    logit = 0.0
    for item in stage.get("top_candidates", []):
        if str(item.get("label", "")).strip().lower() == label:
            probability = float(item.get("confidence", 0.0))
            break
    for item in stage.get("top_logits", []):
        if str(item.get("label", "")).strip().lower() == label:
            logit = float(item.get("logit", 0.0))
            break
    return probability, logit


def main() -> None:
    args = build_parser().parse_args()
    session_dir = Path(args.session_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    gesture_profile_path = Path(args.gesture_profile).resolve()
    output_path = Path(args.output_json).resolve()

    want_alignment_row = _read_target_alignment_row(session_dir, "want")
    want_trigger_row = _read_trigger_row_by_segment_id(session_dir, int(want_alignment_row["segment_id"]))
    cache_path = cache_dir / "continuous_feature_cache_mirror1.npz"
    _metadata, arrays = load_continuous_feature_cache(cache_path)
    serving_records = _records_from_trigger_row(trigger_row=want_trigger_row, arrays=arrays)
    frame_count = len(serving_records)

    results: list[dict[str, object]] = []
    best_row = None
    for anchor in ("torso_center", "mid_shoulder"):
        engine = _configure_engine(
            checkpoint_path=checkpoint_path,
            gesture_profile_path=gesture_profile_path,
            enable_mother_fix=True,
            enable_you_i_fix=True,
        )
        engine.pose_local_anchor = anchor
        engine.enable_you_like_pairwise_calibration = True
        engine.you_like_pairwise_delta = 0.98
        engine.enable_like_i_pairwise_calibration = True
        engine.like_i_pairwise_delta = 2.78
        feature_rows = engine._build_feature_rows_from_records(serving_records, pose_local_anchor=anchor)
        for variant_name, sampled_index_list in _sample_variants(frame_count):
            sequence, build_debug = engine._materialize_trigger_sequence(
                frames=serving_records,
                feature_rows=feature_rows,
                sampled_index_list=sampled_index_list,
                pose_local_anchor=anchor,
            )
            stage = _score_stage(engine=engine, sequence=sequence)
            want_probability, want_logit = _label_score(stage, "want")
            student_probability, student_logit = _label_score(stage, "student")
            row = {
                "anchor": anchor,
                "variant": variant_name,
                "sampled_indices": sampled_index_list,
                "sampled_frame_indices": build_debug["sampled_frame_indices"],
                "raw_label": stage["raw_label"],
                "raw_confidence": stage["raw_confidence"],
                "top_margin": stage["top_margin"],
                "top_candidates": stage["top_candidates"],
                "top_logits": stage["top_logits"],
                "want_probability": _safe_float(want_probability),
                "student_probability": _safe_float(student_probability),
                "want_logit": _safe_float(want_logit),
                "student_logit": _safe_float(student_logit),
                "want_minus_student_logit_gap": _safe_float(want_logit - student_logit),
            }
            results.append(row)
            if best_row is None or (
                float(row["want_probability"]) > float(best_row["want_probability"])
                or (
                    abs(float(row["want_probability"]) - float(best_row["want_probability"])) <= 1e-8
                    and float(row["want_minus_student_logit_gap"]) > float(best_row["want_minus_student_logit_gap"])
                )
            ):
                best_row = row

    output_payload = {
        "session_dir": str(session_dir),
        "cache_path": str(cache_path),
        "target_segment": {
            "segment_id": int(want_alignment_row["segment_id"]),
            "reference_overlap_ratio": _safe_float(float(want_alignment_row["reference_overlap_ratio"])),
            "decision_status": str(want_alignment_row["decision_status"]),
            "raw_label": str(want_alignment_row["raw_label"]),
        },
        "best_variant_by_want_probability": best_row,
        "rows": results,
        "conclusion": (
            "This sweep tests whether the live serving want segment is a near-window problem. "
            "If the best anchor/window variant still leaves want far below student, then the current live want line is not recoverable by simple trigger-local resampling."
        ),
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
