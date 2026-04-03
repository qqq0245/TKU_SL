from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.large_scale_pipeline import iter_video_files, resolve_video_source_root, write_csv
from src.dataset.pipeline_state import refresh_pipeline_state
from src.utils.paths import ensure_dir


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Scan raw_videos dataset and build video manifest.")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--label-map-json", default=str(CONFIG.manifests_dir / "label_map.json"))
    return parser


def load_source_metadata(source_root: Path) -> dict[str, dict]:
    metadata_path = source_root / "source_metadata.csv"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(Path(row["video_path"]).resolve()): row for row in rows if row.get("video_path")}


def main() -> None:
    args = build_parser().parse_args()
    source_root = resolve_video_source_root(Path(args.source_root) if args.source_root else None)
    label_map_path = Path(args.label_map_json)
    with label_map_path.open("r", encoding="utf-8") as handle:
        label_map = json.load(handle)
    labels = label_map["labels"]
    by_label = {row["english_label"]: row for row in labels}

    ensure_dir(CONFIG.manifests_dir)
    ensure_dir(CONFIG.dataset_logs_dir)
    manifest_rows = []
    invalid_rows = []
    source_metadata = load_source_metadata(source_root)

    for label_row in labels:
        english_label = label_row["english_label"]
        label_dir = source_root / english_label
        if not label_dir.exists():
            invalid_rows.append(
                {
                    "english_label": english_label,
                    "class_id": label_row["class_id"],
                    "issue": "missing_label_directory",
                    "path": str(label_dir),
                }
            )
            continue
        video_paths = list(iter_video_files(label_dir))
        if not video_paths:
            invalid_rows.append(
                {
                    "english_label": english_label,
                    "class_id": label_row["class_id"],
                    "issue": "empty_label_directory",
                    "path": str(label_dir),
                }
            )
            continue
        for video_path in video_paths:
            metadata = source_metadata.get(str(video_path.resolve()), {})
            manifest_rows.append(
                {
                    "video_path": str(video_path.resolve()),
                    "english_label": english_label,
                    "class_id": label_row["class_id"],
                    "zh_tw_translation": label_row["zh_tw_translation"],
                    "split_candidate": "pending_split",
                    "file_exists": True,
                    "source_origin_path": metadata.get("source_origin_path", ""),
                    "source_group_key": metadata.get("source_group_key", ""),
                }
            )

    extra_dirs = sorted(path for path in source_root.iterdir() if path.is_dir() and path.name not in by_label)
    for extra_dir in extra_dirs:
        invalid_rows.append(
            {
                "english_label": extra_dir.name,
                "class_id": "",
                "issue": "label_not_in_vocabulary",
                "path": str(extra_dir.resolve()),
            }
        )

    manifest_path = CONFIG.manifests_dir / "video_manifest.csv"
    invalid_path = CONFIG.dataset_logs_dir / "missing_or_invalid_videos.csv"
    write_csv(
        manifest_path,
        [
            "video_path",
            "english_label",
            "class_id",
            "zh_tw_translation",
            "split_candidate",
            "file_exists",
            "source_origin_path",
            "source_group_key",
        ],
        manifest_rows,
    )
    write_csv(invalid_path, ["english_label", "class_id", "issue", "path"], invalid_rows)
    refresh_pipeline_state(
        {
            "script": "scan_video_dataset.py",
            "scanned_videos": len(manifest_rows),
            "invalid_items": len(invalid_rows),
        }
    )

    print(f"Source root: {source_root}")
    print(f"Video manifest: {manifest_path}")
    print(f"Invalid log: {invalid_path}")
    print(f"Valid videos: {len(manifest_rows)}")
    print(f"Issues: {len(invalid_rows)}")


if __name__ == "__main__":
    main()
