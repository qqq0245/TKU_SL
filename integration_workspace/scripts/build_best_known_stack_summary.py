from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from pathlib import Path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize the current best-known serving stack and rank unresolved residuals.")
    parser.add_argument("--baseline-session-dir", required=True)
    parser.add_argument("--best-session-dir", required=True)
    parser.add_argument("--exact-classification-json", required=True)
    parser.add_argument("--teacher-father-analysis-json", required=True)
    parser.add_argument("--teacher-sweep-json", required=True)
    parser.add_argument("--want-sweep-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _token_row(payload: dict[str, object], token: str) -> dict[str, object]:
    for row in payload.get("classifications", []):
        if int(row.get("mirror_input", 0)) != 1:
            continue
        if str(row.get("token", "")).strip().lower() == token:
            return row
    return {}


def _segment_rows(rows: list[dict[str, str]], token: str) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get("reference_label", "")).strip().lower() == token]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _pick_locked_fix_stack(decoder_params: dict[str, object]) -> dict[str, object]:
    keys = (
        "pose_local_anchor",
        "enable_mother_nose_z_calibration",
        "mother_nose_z_offset",
        "enable_you_i_leftbody_calibration",
        "you_i_leftbody_y_offset",
        "enable_you_like_pairwise_calibration",
        "you_like_pairwise_delta",
        "enable_like_i_pairwise_calibration",
        "like_i_pairwise_delta",
        "enable_want_anchor_fallback",
        "want_anchor_fallback_max_nosign_confidence",
        "enable_father_trigger_rescue",
        "father_trigger_rescue_pairwise_delta",
        "father_trigger_rescue_min_ratio",
        "father_trigger_rescue_max_nosign_confidence",
    )
    return {key: decoder_params.get(key) for key in keys if key in decoder_params}


def main() -> None:
    args = build_parser().parse_args()
    baseline_session_dir = Path(args.baseline_session_dir).resolve()
    best_session_dir = Path(args.best_session_dir).resolve()
    exact_json_path = Path(args.exact_classification_json).resolve()
    teacher_father_path = Path(args.teacher_father_analysis_json).resolve()
    teacher_sweep_path = Path(args.teacher_sweep_json).resolve()
    want_sweep_path = Path(args.want_sweep_json).resolve()
    output_path = Path(args.output_json).resolve()

    baseline_summary = _load_json(baseline_session_dir / "session_summary.json")
    best_summary = _load_json(best_session_dir / "session_summary.json")
    exact_payload = _load_json(exact_json_path)
    teacher_father = _load_json(teacher_father_path)
    teacher_sweep = _load_json(teacher_sweep_path)
    want_sweep = _load_json(want_sweep_path)
    best_alignment_rows = _load_csv(best_session_dir / "trigger_segment_alignment.csv")

    baseline_eval = baseline_summary.get("continuous_evaluation", {})
    best_eval = best_summary.get("continuous_evaluation", {})
    decoder_params = best_summary.get("decoder_params", {})

    teacher_exact = _token_row(exact_payload, "teacher")
    want_exact = _token_row(exact_payload, "want")
    i_exact = _token_row(exact_payload, "i")

    teacher_best_variant = teacher_sweep.get("best_teacher_variant", {})
    want_best_variant = want_sweep.get("best_variant_by_want_probability", {})
    teacher_segments = _segment_rows(best_alignment_rows, "teacher")
    want_segments = _segment_rows(best_alignment_rows, "want")

    current_predicted_tokens = list(best_eval.get("predicted_tokens", []))
    locked_fix_validation = {
        "mother_present": "mother" in current_predicted_tokens,
        "you_present": "you" in current_predicted_tokens,
        "like_present": "like" in current_predicted_tokens,
        "father_present": "father" in current_predicted_tokens,
    }

    want_priority_score = (
        _safe_int(want_exact.get("active_frame_count"))
        + sum(_safe_int(row.get("trimmed_length")) for row in want_segments)
    )
    teacher_priority_score = (
        _safe_int(teacher_exact.get("active_frame_count"))
        + sum(_safe_int(row.get("trimmed_length")) for row in teacher_segments)
    )

    output_payload = {
        "baseline_session_dir": str(baseline_session_dir),
        "best_session_dir": str(best_session_dir),
        "best_known_stack": {
            "decoder_params": _pick_locked_fix_stack(decoder_params),
            "reference_tokens": best_eval.get("reference_tokens", []),
            "predicted_tokens": current_predicted_tokens,
            "baseline_word_error_rate": _safe_float(baseline_eval.get("word_error_rate")),
            "current_word_error_rate": _safe_float(best_eval.get("word_error_rate")),
            "word_error_rate_delta": round(
                _safe_float(baseline_eval.get("word_error_rate")) - _safe_float(best_eval.get("word_error_rate")),
                6,
            ),
            "locked_fix_validation": locked_fix_validation,
            "notes": [
                "mother, you/i, you-vs-like, like-vs-i fixes remain locked.",
                "father trigger rescue is part of the current best-known stack.",
                "want anchor fallback remains enabled but has no evidence of solving the live want/student branch.",
            ],
        },
        "unresolved_residuals": {
            "want": {
                "priority_score": want_priority_score,
                "exact_span_predicted_label": want_exact.get("predicted_label"),
                "exact_span_top3": want_exact.get("top3", []),
                "serving_segments": [
                    {
                        "segment_id": _safe_int(row.get("segment_id")),
                        "decision_status": str(row.get("decision_status", "")),
                        "raw_label": str(row.get("raw_label", "")),
                        "trimmed_length": _safe_int(row.get("trimmed_length")),
                        "top_margin": _safe_float(row.get("top_margin")),
                    }
                    for row in want_segments
                ],
                "best_narrow_variant": {
                    "raw_label": want_best_variant.get("raw_label"),
                    "want_probability": _safe_float(want_best_variant.get("want_probability")),
                    "student_probability": _safe_float(want_best_variant.get("student_probability")),
                    "want_minus_student_logit_gap": _safe_float(want_best_variant.get("want_minus_student_logit_gap")),
                },
                "status": "broader_model_domain_investigation",
                "reason": (
                    "Exact-span already collapses to no_sign, live serving emits student on the main want segment, "
                    "and the best narrow window/anchor sweep still leaves want far below student."
                ),
            },
            "teacher": {
                "priority_score": teacher_priority_score,
                "exact_span_predicted_label": teacher_exact.get("predicted_label"),
                "exact_span_top3": teacher_exact.get("top3", []),
                "serving_segments": [
                    {
                        "segment_id": _safe_int(row.get("segment_id")),
                        "decision_status": str(row.get("decision_status", "")),
                        "raw_label": str(row.get("raw_label", "")),
                        "trimmed_length": _safe_int(row.get("trimmed_length")),
                        "top_margin": _safe_float(row.get("top_margin")),
                    }
                    for row in teacher_segments
                ],
                "best_narrow_variant": {
                    "predicted_label": teacher_best_variant.get("predicted_label"),
                    "teacher_probability": _safe_float(teacher_best_variant.get("teacher_probability")),
                    "student_probability": _safe_float(teacher_best_variant.get("student_probability")),
                    "teacher_minus_nosign_logit_gap": _safe_float(teacher_best_variant.get("teacher_minus_nosign_logit_gap")),
                },
                "status": "broader_model_domain_investigation",
                "reason": (
                    "Exact-span is already deeply no_sign and no narrow window/anchor variant recovers teacher "
                    "from the current serving student/no_sign split."
                ),
            },
            "i": {
                "priority_score": _safe_int(i_exact.get("active_frame_count")),
                "exact_span_predicted_label": i_exact.get("predicted_label"),
                "exact_span_top3": i_exact.get("top3", []),
                "status": "not_current_serving_mainline",
                "reason": (
                    "The current progress log treats the you/i family fix as locked. The remaining i miss is not the "
                    "highest-priority serving residual on current evidence, so it stays below the broader want/teacher branch."
                ),
            },
        },
        "father_rescue_status": {
            "current_word_error_rate": _safe_float(teacher_father.get("baseline_word_error_rate")),
            "counterfactual_word_error_rate": teacher_father.get("counterfactual_word_error_rate", {}),
            "recommendation": teacher_father.get("recommendation", {}),
            "status": "locked_improvement",
        },
        "residual_priority_order": [
            {
                "token": "want",
                "priority_score": want_priority_score,
                "why": "No narrow fix remains; larger active support than teacher on current evidence.",
            },
            {
                "token": "teacher",
                "priority_score": teacher_priority_score,
                "why": "No narrow fix remains, but support is smaller than want on the current session.",
            },
            {
                "token": "i",
                "priority_score": _safe_int(i_exact.get("active_frame_count")),
                "why": "Still unresolved at exact-span, but not the highest-priority current serving residual.",
            },
        ],
        "recommended_next_mainline": {
            "token": "want",
            "family": "broader_model_domain_investigation",
            "reason": (
                "After father is locked and teacher narrow repair paths are downgraded, the highest-value unresolved "
                "residual is want: it still fails at exact-span and live serving, and its narrow window path has already "
                "been rigorously ruled out."
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
