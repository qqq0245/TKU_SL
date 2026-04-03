from __future__ import annotations

import json
import random
import re
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.batch_utils import write_csv_rows
from src.dataset.pipeline_state import refresh_pipeline_state


SPLIT_FIELDS = [
    "sample_id",
    "sample_path",
    "class_id",
    "english_label",
    "zh_tw_translation",
    "feature_mode",
    "feature_dim",
    "sequence_length",
    "source_video_path",
    "source_group_key",
]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build train/val/test splits from processed sequences.")
    parser.add_argument("--index-jsonl", default=str(CONFIG.processed_sequences_dir / "index.jsonl"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=CONFIG.random_seed)
    parser.add_argument("--output-dir", default=str(CONFIG.splits_dir))
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    return parser


def _is_usable_sample(sample_path: Path) -> bool:
    data = np.load(sample_path, allow_pickle=True)
    if "frame_valid_mask" not in data:
        return True
    frame_valid = data["frame_valid_mask"].astype(np.float32)
    if frame_valid.ndim == 1:
        valid_ratio = float((frame_valid > 0).mean())
    else:
        valid_ratio = float((frame_valid.sum(axis=1) > 0).mean())
    return valid_ratio >= CONFIG.min_valid_frame_ratio


_AUGMENT_SUFFIX_PATTERN = re.compile(r"_(orig|bright|dim|flip)$", re.IGNORECASE)


def _canonicalize_group_path(path_value: str) -> str:
    path = Path(path_value)
    stem = _AUGMENT_SUFFIX_PATTERN.sub("", path.stem)
    try:
        return str(path.with_name(f"{stem}{path.suffix}").resolve())
    except OSError:
        return str(path.with_name(f"{stem}{path.suffix}"))


def _build_source_group_key(record: dict, sample_path: Path) -> str:
    metadata = record.get("metadata", {})
    explicit_group_key = str(metadata.get("source_group_key") or "").strip()
    if explicit_group_key:
        return _canonicalize_group_path(explicit_group_key)
    source_video_path = str(metadata.get("source_video_path") or "").strip()
    if source_video_path:
        return _canonicalize_group_path(source_video_path)
    sample_id = str(record.get("sample_id") or "").strip()
    if sample_id:
        return f"sample_id::{sample_id}"
    return f"sample_path::{sample_path.resolve()}"


def _assign_grouped_rows(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    active_splits = ["train"]
    if val_ratio > 0:
        active_splits.append("val")
    if test_ratio > 0:
        active_splits.append("test")

    grouped_rows: dict[str, list[dict]] = {}
    for row in rows:
        grouped_rows.setdefault(row["source_group_key"], []).append(row)

    group_items = list(grouped_rows.items())
    groups = []
    for group_key, group_rows in group_items:
        label_counts = Counter(row["english_label"] for row in group_rows)
        groups.append(
            {
                "group_key": group_key,
                "rows": group_rows,
                "size": len(group_rows),
                "label_counts": label_counts,
            }
        )
    rng.shuffle(groups)
    groups.sort(key=lambda item: (item["size"], len(item["label_counts"])), reverse=True)

    group_count = len(groups)
    if group_count == 1:
        return list(groups[0]["rows"]), [], []

    total_rows = len(rows)
    label_totals = Counter(row["english_label"] for row in rows)

    target_rows = {
        "train": float(total_rows * train_ratio),
        "val": float(total_rows * val_ratio),
        "test": float(total_rows * test_ratio),
    }
    target_labels = {
        split_name: {
            label: float(total_count * ratio)
            for label, total_count in label_totals.items()
        }
        for split_name, ratio in (
            ("train", train_ratio),
            ("val", val_ratio),
            ("test", test_ratio),
        )
    }
    required_groups = {
        "train": 1,
        "val": 1 if group_count >= 3 and val_ratio > 0 else 0,
        "test": 1 if group_count >= 3 and test_ratio > 0 else 0,
    }
    assigned_rows = {"train": 0, "val": 0, "test": 0}
    assigned_groups = {"train": 0, "val": 0, "test": 0}
    split_rows = {"train": [], "val": [], "test": []}
    split_label_counts = {name: Counter() for name in ("train", "val", "test")}

    for index, group in enumerate(groups):
        remaining_groups_after = group_count - index - 1
        missing_required = {
            name: max(0, required_groups[name] - assigned_groups[name]) for name in ("train", "val", "test")
        }
        forced_splits = [
            name
            for name in active_splits
            if missing_required[name] > 0 and remaining_groups_after == sum(missing_required.values()) - 1
        ]
        candidate_splits = forced_splits or [
            name
            for name in active_splits
            if remaining_groups_after >= (sum(missing_required.values()) - missing_required[name])
        ]
        if not candidate_splits:
            candidate_splits = list(active_splits)

        def split_priority(name: str) -> tuple[int, float, float, float, int]:
            still_required = 1 if assigned_groups[name] < required_groups[name] else 0
            total_need = target_rows[name] - assigned_rows[name]
            label_need = sum(
                target_labels[name].get(label, 0.0) - split_label_counts[name].get(label, 0)
                for label in group["label_counts"]
            )
            overshoot = max(0.0, group["size"] - max(total_need, 0.0))
            return (still_required, label_need, total_need, -overshoot, -assigned_rows[name])

        chosen_split = max(candidate_splits, key=split_priority)
        split_rows[chosen_split].extend(group["rows"])
        assigned_rows[chosen_split] += group["size"]
        assigned_groups[chosen_split] += 1
        split_label_counts[chosen_split].update(group["label_counts"])

    return split_rows["train"], split_rows["val"], split_rows["test"]


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)

    rows: list[dict] = []
    class_counts: Counter[int] = Counter()
    low_sample_rows = []
    with Path(args.index_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            metadata = record.get("metadata", {})
            sample_path = (Path(args.index_jsonl).parent / record["path"]).resolve()
            if not _is_usable_sample(sample_path):
                continue
            row = {
                "sample_id": record["sample_id"],
                "sample_path": str(sample_path),
                "class_id": record.get("class_id", metadata.get("class_id")),
                "english_label": record["class_label"],
                "zh_tw_translation": metadata.get("zh_tw_translation", ""),
                "feature_mode": record.get("feature_mode") or metadata.get("feature_mode"),
                "feature_dim": record.get("feature_dim"),
                "sequence_length": record.get("sequence_length"),
                "source_video_path": metadata.get("source_video_path", ""),
                "source_group_key": _build_source_group_key(record, sample_path),
            }
            rows.append(row)
            class_counts[int(row["class_id"])] += 1

    label_lookup = {}
    for row in rows:
        label_lookup[int(row["class_id"])] = row["english_label"]

    for class_id, total in sorted(class_counts.items()):
        if total < args.min_samples_per_class:
            low_sample_rows.append(
                {
                    "class_id": class_id,
                    "english_label": label_lookup.get(class_id, ""),
                    "usable_sequence_count": total,
                    "issue": "low_sample",
                }
            )

    train_rows, val_rows, test_rows = _assign_grouped_rows(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        rng=rng,
    )

    write_csv_rows(output_dir / "train.csv", SPLIT_FIELDS, train_rows)
    write_csv_rows(output_dir / "val.csv", SPLIT_FIELDS, val_rows)
    write_csv_rows(output_dir / "test.csv", SPLIT_FIELDS, test_rows)
    write_csv_rows(
        output_dir / "low_sample_classes.csv",
        ["class_id", "english_label", "usable_sequence_count", "issue"],
        low_sample_rows,
    )

    refresh_pipeline_state(
        {
            "script": "build_splits.py",
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "low_sample_class_count": len(low_sample_rows),
        }
    )

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Low sample classes: {len(low_sample_rows)}")
    print(f"Split dir: {output_dir}")


if __name__ == "__main__":
    main()
