from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.large_scale_pipeline import build_label_rows, write_csv, write_json
from src.dataset.pipeline_state import refresh_pipeline_state
from src.utils.paths import ensure_dir


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build label map and normalized vocabulary manifest.")
    parser.add_argument("--vocabulary-csv", default=str(CONFIG.vocabulary_list_csv))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    vocabulary_csv = Path(args.vocabulary_csv)
    rows = build_label_rows(vocabulary_csv)

    ensure_dir(CONFIG.manifests_dir)
    label_map_path = CONFIG.manifests_dir / "label_map.json"
    vocabulary_manifest_path = CONFIG.manifests_dir / "vocabulary_manifest.csv"

    write_json(
        label_map_path,
        {
            "source_csv": str(vocabulary_csv),
            "class_count": len(rows),
            "labels": rows,
            "by_english_label": {row["english_label"]: row for row in rows},
        },
    )
    write_csv(
        vocabulary_manifest_path,
        ["class_id", "english_label", "label_slug", "zh_tw_translation", "video_count"],
        rows,
    )
    refresh_pipeline_state(
        {
            "script": "build_label_map.py",
            "label_count": len(rows),
        }
    )
    print(f"Saved label map to {label_map_path}")
    print(f"Saved vocabulary manifest to {vocabulary_manifest_path}")
    print(f"Class count: {len(rows)}")


if __name__ == "__main__":
    main()
