from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.batch_utils import read_csv_rows, write_csv_rows
from src.dataset.pipeline_state import refresh_pipeline_state


def main() -> None:
    with (CONFIG.manifests_dir / "label_map.json").open("r", encoding="utf-8") as handle:
        label_rows = json.load(handle).get("labels", [])

    manifest_rows = read_csv_rows(CONFIG.manifests_dir / "video_manifest.csv")
    landmarks_failed_rows = read_csv_rows(CONFIG.dataset_logs_dir / "landmarks_failed.csv")
    sequences_failed_rows = read_csv_rows(CONFIG.dataset_logs_dir / "sequences_failed.csv")
    train_rows = read_csv_rows(CONFIG.splits_dir / "train.csv")
    val_rows = read_csv_rows(CONFIG.splits_dir / "val.csv")
    test_rows = read_csv_rows(CONFIG.splits_dir / "test.csv")
    low_sample_rows = read_csv_rows(CONFIG.splits_dir / "low_sample_classes.csv")
    low_sample_ids = {str(row["class_id"]) for row in low_sample_rows}

    scanned = defaultdict(int)
    landmarks_success = defaultdict(int)
    landmarks_failed = defaultdict(int)
    sequence_success = defaultdict(int)
    sequence_failed = defaultdict(int)
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    test_counts = defaultdict(int)
    label_to_class = {row["english_label"]: str(row["class_id"]) for row in label_rows}

    for row in manifest_rows:
        scanned[str(row["class_id"])] += 1
    for cache in CONFIG.landmarks_cache_dir.rglob("*.npz"):
        class_id = label_to_class.get(cache.parent.name)
        if class_id is not None:
            landmarks_success[class_id] += 1
    for row in landmarks_failed_rows:
        landmarks_failed[str(row["class_id"])] += 1
    index_path = CONFIG.processed_sequences_dir / "index.jsonl"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    class_id = record.get("class_id", record.get("metadata", {}).get("class_id"))
                    if class_id is not None:
                        sequence_success[str(class_id)] += 1
    for row in sequences_failed_rows:
        sequence_failed[str(row["class_id"])] += 1
    for row in train_rows:
        train_counts[str(row["class_id"])] += 1
    for row in val_rows:
        val_counts[str(row["class_id"])] += 1
    for row in test_rows:
        test_counts[str(row["class_id"])] += 1

    coverage_rows = []
    summary = {"class_count": len(label_rows), "split_ready_classes": 0, "low_sample_classes": 0, "no_video_classes": 0}
    for row in label_rows:
        class_id = str(row["class_id"])
        usable = sequence_success.get(class_id, 0)
        if scanned.get(class_id, 0) == 0:
            status = "no_video"
            summary["no_video_classes"] += 1
        elif landmarks_success.get(class_id, 0) == 0:
            status = "scanned_only"
        elif train_counts.get(class_id, 0) + val_counts.get(class_id, 0) + test_counts.get(class_id, 0) > 0 and usable >= 2:
            status = "split_ready"
            summary["split_ready_classes"] += 1
        elif class_id in low_sample_ids or usable < 2:
            status = "low_sample"
            summary["low_sample_classes"] += 1
        elif sequence_failed.get(class_id, 0) > 0:
            status = "sequence_partial"
        elif landmarks_failed.get(class_id, 0) > 0:
            status = "landmarks_partial"
        else:
            status = "sequence_partial"

        coverage_rows.append(
            {
                "class_id": row["class_id"],
                "english_label": row["english_label"],
                "zh_tw_translation": row.get("zh_tw_translation", ""),
                "declared_video_count": row.get("video_count", 0),
                "scanned_video_count": scanned.get(class_id, 0),
                "landmarks_success_count": landmarks_success.get(class_id, 0),
                "landmarks_failed_count": landmarks_failed.get(class_id, 0),
                "sequence_success_count": usable,
                "sequence_failed_count": sequence_failed.get(class_id, 0),
                "usable_sequence_count": usable,
                "train_count": train_counts.get(class_id, 0),
                "val_count": val_counts.get(class_id, 0),
                "test_count": test_counts.get(class_id, 0),
                "coverage_status": status,
            }
        )

    write_csv_rows(
        CONFIG.dataset_logs_dir / "class_coverage_report.csv",
        list(coverage_rows[0].keys()) if coverage_rows else [],
        coverage_rows,
    )
    with (CONFIG.dataset_logs_dir / "class_coverage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    refresh_pipeline_state({"script": "build_class_coverage_report.py"})
    print(f"Coverage report: {CONFIG.dataset_logs_dir / 'class_coverage_report.csv'}")
    print(f"Coverage summary: {CONFIG.dataset_logs_dir / 'class_coverage_summary.json'}")


if __name__ == "__main__":
    main()
