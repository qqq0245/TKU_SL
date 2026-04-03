from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.batch_utils import read_csv_rows, write_csv_rows


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build subset experiment from split files and coverage report.")
    parser.add_argument("--split-dir", default=str(CONFIG.splits_dir))
    parser.add_argument("--output-name", default="subset_experiment")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--class-ids", nargs="*", default=None)
    parser.add_argument("--min-samples-per-class", type=int, default=1)
    parser.add_argument("--min-usable-sequences", type=int, default=1)
    parser.add_argument(
        "--sort-by",
        default="usable_sequence_count",
        choices=["usable_sequence_count", "train_count", "declared_video_count"],
    )
    parser.add_argument("--balanced-per-class", action="store_true")
    return parser


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "sample_id",
        "sample_path",
        "class_id",
        "english_label",
        "zh_tw_translation",
        "feature_mode",
        "feature_dim",
        "sequence_length",
        "source_video_path",
    ]
    write_csv_rows(path, fields, rows)


def main() -> None:
    args = build_parser().parse_args()
    split_dir = Path(args.split_dir)
    train_rows = read_csv_rows(split_dir / "train.csv")
    val_rows = read_csv_rows(split_dir / "val.csv")
    test_rows = read_csv_rows(split_dir / "test.csv")
    all_rows = train_rows + val_rows + test_rows
    class_counts = Counter(row["english_label"] for row in all_rows)
    coverage_rows = read_csv_rows(CONFIG.dataset_logs_dir / "class_coverage_report.csv")
    coverage_by_label = {row["english_label"]: row for row in coverage_rows}

    if args.class_ids:
        selected_labels = {
            row["english_label"] for row in coverage_rows if str(row.get("class_id")) in {str(x) for x in args.class_ids}
        }
    elif args.labels:
        selected_labels = set(args.labels)
    else:
        candidates = []
        for label, count in class_counts.items():
            coverage = coverage_by_label.get(label, {})
            usable = int(coverage.get("usable_sequence_count", count))
            if count < args.min_samples_per_class or usable < args.min_usable_sequences:
                continue
            score = int(coverage.get(args.sort_by, usable))
            candidates.append((label, score, usable))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        cap = args.max_classes or args.top_k or 20
        selected_labels = {label for label, _, _ in candidates[:cap]}

    def filter_rows(rows: list[dict]) -> list[dict]:
        filtered = [row for row in rows if row["english_label"] in selected_labels]
        if not args.balanced_per_class:
            return filtered
        counts = Counter(row["english_label"] for row in filtered)
        if not counts:
            return filtered
        target = min(counts.values())
        used = Counter()
        balanced = []
        for row in filtered:
            label = row["english_label"]
            if used[label] < target:
                balanced.append(row)
                used[label] += 1
        return balanced

    subset_train = filter_rows(train_rows)
    subset_val = filter_rows(val_rows)
    subset_test = filter_rows(test_rows)
    subset_dir = split_dir / "subsets" / args.output_name
    _write_rows(subset_dir / "train.csv", subset_train)
    _write_rows(subset_dir / "val.csv", subset_val)
    _write_rows(subset_dir / "test.csv", subset_test)

    manifest_rows = []
    for label in sorted(selected_labels):
        coverage = coverage_by_label.get(label, {})
        manifest_rows.append(
            {
                "english_label": label,
                "sample_count": class_counts[label],
                "usable_sequence_count": coverage.get("usable_sequence_count", class_counts[label]),
                "coverage_status": coverage.get("coverage_status", "unknown"),
            }
        )
    write_csv_rows(
        subset_dir / "subset_manifest.csv",
        ["english_label", "sample_count", "usable_sequence_count", "coverage_status"],
        manifest_rows,
    )
    subset_coverage_rows = [row for row in coverage_rows if row["english_label"] in selected_labels]
    if subset_coverage_rows:
        write_csv_rows(subset_dir / "subset_coverage_report.csv", list(subset_coverage_rows[0].keys()), subset_coverage_rows)
    with (subset_dir / "subset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_class_count": len(selected_labels),
                "selected_labels": sorted(selected_labels),
                "train_rows": len(subset_train),
                "val_rows": len(subset_val),
                "test_rows": len(subset_test),
                "min_usable_sequences": args.min_usable_sequences,
                "sort_by": args.sort_by,
                "balanced_per_class": args.balanced_per_class,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Subset dir: {subset_dir}")
    print(f"Selected classes: {len(selected_labels)}")
    print(f"Train rows: {len(subset_train)}")
    print(f"Val rows: {len(subset_val)}")
    print(f"Test rows: {len(subset_test)}")


if __name__ == "__main__":
    main()
