from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.batch_utils import read_csv_rows
from src.dataset.pipeline_state import refresh_pipeline_state


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Show current pipeline status and optional save to file.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--txt-out", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = refresh_pipeline_state({"script": "pipeline_status.py"})
    coverage_rows = read_csv_rows(CONFIG.dataset_logs_dir / "class_coverage_report.csv")
    split_ready = [row for row in coverage_rows if row.get("coverage_status") == "split_ready"]
    low_coverage = [row for row in coverage_rows if row.get("coverage_status") in {"no_video", "low_sample", "scanned_only"}]
    top_classes = sorted(coverage_rows, key=lambda row: int(row.get("usable_sequence_count", 0)), reverse=True)[:5]

    lines = [
        f"labels: {state.get('label_count', 0)}",
        f"videos scanned: {state.get('scanned_videos', 0)}",
        f"landmarks success: {state.get('landmarks_extracted_count', 0)}",
        f"landmarks failed: {state.get('landmarks_failed_count', 0)}",
        f"sequences exported: {state.get('sequences_exported_count', 0)}",
        f"sequences failed: {state.get('sequences_failed_count', 0)}",
        f"split ready classes: {len(split_ready)}",
        f"low coverage classes: {len(low_coverage)}",
        f"last updated: {state.get('last_updated', '')}",
        "top classes by usable sequences:",
    ]
    for row in top_classes:
        lines.append(f"  - {row.get('english_label')} ({row.get('usable_sequence_count')} usable, status={row.get('coverage_status')})")
    output = "\n".join(lines)
    print(output)
    if args.txt_out:
        Path(args.txt_out).write_text(output, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
