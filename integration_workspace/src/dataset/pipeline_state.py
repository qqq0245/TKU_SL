from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from config import CONFIG
from src.dataset.batch_utils import read_csv_rows
from src.utils.paths import ensure_dir


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def _load_index_rows(index_path: Path) -> list[dict]:
    if not index_path.exists():
        return []
    rows = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _count_landmarks_cache() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not CONFIG.landmarks_cache_dir.exists():
        return {}
    for path in CONFIG.landmarks_cache_dir.rglob("*.npz"):
        counts[path.parent.name] += 1
    return dict(counts)


def _count_sequences(index_rows: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    by_label: dict[str, int] = defaultdict(int)
    by_class_id: dict[str, int] = defaultdict(int)
    for row in index_rows:
        label = row.get("class_label", "")
        class_id = row.get("class_id", row.get("metadata", {}).get("class_id"))
        by_label[label] += 1
        if class_id is not None:
            by_class_id[str(class_id)] += 1
    return dict(by_label), dict(by_class_id)


def _count_split_rows(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    class_counts: dict[str, int] = defaultdict(int)
    label_counts: dict[str, int] = defaultdict(int)
    for row in read_csv_rows(path):
        class_counts[str(row.get("class_id", ""))] += 1
        label_counts[row.get("english_label", "")] += 1
    return dict(class_counts), dict(label_counts)


def refresh_pipeline_state(run_summary: dict | None = None) -> dict:
    ensure_dir(CONFIG.dataset_logs_dir)
    state_path = CONFIG.dataset_logs_dir / "pipeline_state.json"
    previous_state = _load_json(state_path)

    label_map = _load_json(CONFIG.manifests_dir / "label_map.json")
    label_rows = label_map.get("labels", [])
    manifest_rows = read_csv_rows(CONFIG.manifests_dir / "video_manifest.csv")
    landmarks_failed_rows = read_csv_rows(CONFIG.dataset_logs_dir / "landmarks_failed.csv")
    sequences_failed_rows = read_csv_rows(CONFIG.dataset_logs_dir / "sequences_failed.csv")
    index_rows = _load_index_rows(CONFIG.processed_sequences_dir / "index.jsonl")
    train_class_counts, _ = _count_split_rows(CONFIG.splits_dir / "train.csv")
    val_class_counts, _ = _count_split_rows(CONFIG.splits_dir / "val.csv")
    test_class_counts, _ = _count_split_rows(CONFIG.splits_dir / "test.csv")

    manifest_by_label: dict[str, int] = defaultdict(int)
    manifest_by_class_id: dict[str, int] = defaultdict(int)
    for row in manifest_rows:
        manifest_by_label[row["english_label"]] += 1
        manifest_by_class_id[str(row["class_id"])] += 1

    landmarks_success_by_label = _count_landmarks_cache()
    landmarks_failed_by_class: dict[str, int] = defaultdict(int)
    for row in landmarks_failed_rows:
        landmarks_failed_by_class[str(row.get("class_id", ""))] += 1

    sequence_success_by_label, sequence_success_by_class = _count_sequences(index_rows)
    sequence_failed_by_class: dict[str, int] = defaultdict(int)
    for row in sequences_failed_rows:
        sequence_failed_by_class[str(row.get("class_id", ""))] += 1

    per_class = []
    for row in label_rows:
        class_id = str(row["class_id"])
        english_label = row["english_label"]
        per_class.append(
            {
                "class_id": row["class_id"],
                "english_label": english_label,
                "zh_tw_translation": row.get("zh_tw_translation", ""),
                "declared_video_count": row.get("video_count", 0),
                "scanned_video_count": manifest_by_class_id.get(class_id, 0),
                "landmarks_extracted_count": landmarks_success_by_label.get(english_label, 0),
                "landmarks_failed_count": landmarks_failed_by_class.get(class_id, 0),
                "sequences_exported_count": sequence_success_by_class.get(class_id, 0),
                "sequences_failed_count": sequence_failed_by_class.get(class_id, 0),
                "train_count": train_class_counts.get(class_id, 0),
                "val_count": val_class_counts.get(class_id, 0),
                "test_count": test_class_counts.get(class_id, 0),
            }
        )

    skipped_existing_count = int(previous_state.get("skipped_existing_count", 0))
    if run_summary:
        skipped_existing_count += int(run_summary.get("skipped_existing_count", 0))

    state = {
        "label_count": len(label_rows),
        "total_videos": len(manifest_rows),
        "scanned_videos": len(manifest_rows),
        "landmarks_extracted_count": sum(item["landmarks_extracted_count"] for item in per_class),
        "landmarks_failed_count": len(landmarks_failed_rows),
        "sequences_exported_count": len(index_rows),
        "sequences_failed_count": len(sequences_failed_rows),
        "skipped_existing_count": skipped_existing_count,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_run": run_summary or previous_state.get("last_run", {}),
        "per_class_processed_counts": per_class,
    }

    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    return state


__all__ = ["refresh_pipeline_state"]
