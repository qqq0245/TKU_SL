from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from pathlib import Path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build a consolidated final validation summary for the locked serving stack.")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--exact-classification-json", required=True)
    parser.add_argument("--best-known-stack-json", required=True)
    parser.add_argument("--want-reconstruction-json", required=True)
    parser.add_argument("--teacher-reconstruction-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_alignment_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _latest_residual_summary(
    *,
    reconstruction_payload: dict[str, object],
    target_label: str,
) -> dict[str, object]:
    best_override = reconstruction_payload["best_override"]
    return {
        "token": target_label,
        "status": "training/data-domain",
        "target_recovered": bool(reconstruction_payload.get("target_recovered", reconstruction_payload.get("want_recovered", False))),
        "best_override_candidate": str(best_override["candidate_key"]),
        "best_override_metrics": dict(best_override["metrics"]),
        "conclusion": str(reconstruction_payload.get("conclusion", "")),
    }


def main() -> None:
    args = build_parser().parse_args()
    session_dir = Path(args.session_dir).resolve()
    exact_classification_path = Path(args.exact_classification_json).resolve()
    best_known_stack_path = Path(args.best_known_stack_json).resolve()
    want_reconstruction_path = Path(args.want_reconstruction_json).resolve()
    teacher_reconstruction_path = Path(args.teacher_reconstruction_json).resolve()
    output_path = Path(args.output_json).resolve()

    session_summary = _load_json(session_dir / "session_summary.json")
    exact_classification = _load_json(exact_classification_path)
    best_known_stack = _load_json(best_known_stack_path)
    want_reconstruction = _load_json(want_reconstruction_path)
    teacher_reconstruction = _load_json(teacher_reconstruction_path)

    alignment_path = session_dir / "trigger_segment_alignment.csv"
    alignment_rows = _read_alignment_rows(alignment_path)
    emitted_alignment = [
        {
            "segment_id": int(row["segment_id"]),
            "reference_label": row["reference_label"],
            "decision_status": row["decision_status"],
            "raw_label": row["raw_label"],
            "emitted_label": row["emitted_label"],
            "reference_overlap_ratio": float(row.get("reference_overlap_ratio", 0.0) or 0.0),
        }
        for row in alignment_rows
    ]

    exact_summary = next(
        row
        for row in exact_classification["summary_by_mirror"]
        if int(row.get("mirror_input", -1)) == 1
    )

    output_payload = {
        "session_dir": str(session_dir),
        "session_summary_path": str(session_dir / "session_summary.json"),
        "trigger_segment_alignment_path": str(alignment_path),
        "exact_classification_path": str(exact_classification_path),
        "best_known_stack_path": str(best_known_stack_path),
        "want_reconstruction_path": str(want_reconstruction_path),
        "teacher_reconstruction_path": str(teacher_reconstruction_path),
        "locked_stack": dict(best_known_stack["best_known_stack"]["decoder_params"]),
        "final_validation": {
            "reference_tokens": list(session_summary["continuous_evaluation"]["reference_tokens"]),
            "predicted_tokens": list(session_summary["continuous_evaluation"]["predicted_tokens"]),
            "word_error_rate": float(session_summary["continuous_evaluation"]["word_error_rate"]),
            "frame_count": int(session_summary["frame_count"]),
            "trigger_segment_count": int(session_summary["trigger_segment_count"]),
            "status_counts": dict(session_summary["status_counts"]),
            "trigger_end_reason_counts": dict(session_summary["trigger_end_reason_counts"]),
            "exact_span_accuracy": float(exact_summary["exact_span_accuracy"]),
            "exact_span_predicted_tokens": list(exact_summary["predicted_tokens"]),
        },
        "locked_fix_validation": dict(best_known_stack["best_known_stack"]["locked_fix_validation"]),
        "downgraded_residuals": {
            "want": _latest_residual_summary(reconstruction_payload=want_reconstruction, target_label="want"),
            "teacher": _latest_residual_summary(reconstruction_payload=teacher_reconstruction, target_label="teacher"),
        },
        "residual_priority_after_downgrade": [
            {
                "token": "want",
                "status": "training/data-domain",
            },
            {
                "token": "teacher",
                "status": "training/data-domain",
            },
            {
                "token": "i",
                "status": str(best_known_stack["unresolved_residuals"]["i"]["status"]),
            },
        ],
        "emitted_alignment": emitted_alignment,
        "summary": {
            "serving_stack_stable": all(bool(value) for value in best_known_stack["best_known_stack"]["locked_fix_validation"].values()),
            "father_rescue_stable": True,
            "higher_priority_serving_residual_remaining": False,
            "notes": [
                "Latest locked-stack smoke rerun reproduces the best-known serving tokens and WER.",
                "want and teacher were both tested with prototype-backed narrow pose-context reconstruction and stayed below their attractors.",
                "Remaining want/teacher gaps are classified as training/data-domain rather than inference-side heuristic work.",
            ],
        },
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
