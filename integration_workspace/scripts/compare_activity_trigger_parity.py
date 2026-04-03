from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from pathlib import Path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Compare activity-derived exact-span predictions against trigger-segment parity.")
    parser.add_argument("--activity-json", required=True)
    parser.add_argument("--trigger-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = build_parser().parse_args()
    activity_path = Path(args.activity_json).resolve()
    trigger_path = Path(args.trigger_csv).resolve()
    output_path = Path(args.output_json).resolve()

    activity_payload = json.loads(activity_path.read_text(encoding="utf-8"))
    trigger_rows = _read_csv_rows(trigger_path)

    trigger_by_reference: dict[str, list[dict[str, str]]] = {}
    for row in trigger_rows:
        reference_label = row.get("reference_label", "").strip().lower()
        if not reference_label:
            continue
        trigger_by_reference.setdefault(reference_label, []).append(row)

    comparisons: list[dict[str, object]] = []
    for summary in activity_payload.get("summary_by_mirror", []):
        mirror_input = int(summary["mirror_input"])
        per_token_rows = [
            row
            for row in activity_payload.get("classifications", [])
            if int(row.get("mirror_input", -1)) == mirror_input
        ]
        token_map = {str(row["token"]): row for row in per_token_rows}
        mirror_rows: list[dict[str, object]] = []
        for token in summary.get("reference_tokens", []):
            token_str = str(token)
            activity_row = token_map.get(token_str, {})
            trigger_segments = trigger_by_reference.get(token_str, [])
            mirror_rows.append(
                {
                    "token": token_str,
                    "activity_predicted_label": activity_row.get("predicted_label", ""),
                    "activity_top3": activity_row.get("top3", []),
                    "activity_correct": int(activity_row.get("correct", 0)),
                    "trigger_segment_confusions": [row.get("segment_confusion", "") for row in trigger_segments],
                    "trigger_segment_labels": [row.get("raw_label", "") for row in trigger_segments],
                    "trigger_segment_statuses": [row.get("decision_status", "") for row in trigger_segments],
                }
            )
        comparisons.append(
            {
                "mirror_input": mirror_input,
                "exact_span_accuracy": summary.get("exact_span_accuracy", 0.0),
                "predicted_label_counts": summary.get("predicted_label_counts", {}),
                "per_token": mirror_rows,
            }
        )

    output_payload = {
        "activity_json": str(activity_path),
        "trigger_csv": str(trigger_path),
        "comparisons": comparisons,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
