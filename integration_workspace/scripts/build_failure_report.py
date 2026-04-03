from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.batch_utils import read_csv_rows, write_csv_rows


def main() -> None:
    rows = []
    for row in read_csv_rows(CONFIG.dataset_logs_dir / "missing_or_invalid_videos.csv"):
        rows.append(
            {
                "stage": "scan",
                "english_label": row.get("english_label", ""),
                "class_id": row.get("class_id", ""),
                "target_path": row.get("path", ""),
                "error_type": row.get("issue", ""),
                "error_message": row.get("issue", ""),
            }
        )
    for row in read_csv_rows(CONFIG.dataset_logs_dir / "landmarks_failed.csv"):
        rows.append(
            {
                "stage": "landmarks",
                "english_label": row.get("english_label", ""),
                "class_id": row.get("class_id", ""),
                "target_path": row.get("video_path", ""),
                "error_type": row.get("error", ""),
                "error_message": row.get("traceback", "") or row.get("error", ""),
            }
        )
    for row in read_csv_rows(CONFIG.dataset_logs_dir / "sequences_failed.csv"):
        rows.append(
            {
                "stage": "sequence_export",
                "english_label": row.get("english_label", ""),
                "class_id": row.get("class_id", ""),
                "target_path": row.get("source_video_path", ""),
                "error_type": row.get("error", ""),
                "error_message": row.get("traceback", "") or row.get("error", ""),
            }
        )

    write_csv_rows(
        CONFIG.dataset_logs_dir / "failure_report.csv",
        ["stage", "english_label", "class_id", "target_path", "error_type", "error_message"],
        rows,
    )
    counter = Counter((row["stage"], row["error_type"]) for row in rows)
    summary = {
        "total_failures": len(rows),
        "by_stage": dict(Counter(row["stage"] for row in rows)),
        "top_error_types": [
            {"stage": stage, "error_type": error_type, "count": count}
            for (stage, error_type), count in counter.most_common(20)
        ],
    }
    with (CONFIG.dataset_logs_dir / "failure_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Failure report: {CONFIG.dataset_logs_dir / 'failure_report.csv'}")
    print(f"Failure summary: {CONFIG.dataset_logs_dir / 'failure_summary.json'}")


if __name__ == "__main__":
    main()
