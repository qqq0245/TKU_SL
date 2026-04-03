from __future__ import annotations

import csv
import json
import sys
import traceback
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.batch_utils import filter_rows, merge_failure_rows, read_csv_rows, write_csv_rows
from src.dataset.large_scale_pipeline import extract_video_landmarks
from src.dataset.pipeline_state import refresh_pipeline_state
from src.utils.paths import ensure_dir


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Batch extract MediaPipe landmarks from manifest videos.")
    parser.add_argument("--manifest-csv", default=str(CONFIG.manifests_dir / "video_manifest.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--class-filter", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def _cache_path(video_path: str) -> Path:
    video = Path(video_path)
    label_dir = CONFIG.landmarks_cache_dir / video.parent.name
    ensure_dir(label_dir)
    return label_dir / f"{video.stem}.npz"


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(CONFIG.landmarks_cache_dir)
    ensure_dir(CONFIG.dataset_logs_dir)

    with Path(args.manifest_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    failure_path = CONFIG.dataset_logs_dir / "landmarks_failed.csv"
    existing_failure_rows = read_csv_rows(failure_path)
    rows = manifest_rows
    if args.retry_failed:
        failed_paths = {row["video_path"] for row in existing_failure_rows}
        rows = [row for row in manifest_rows if row["video_path"] in failed_paths]
    rows = filter_rows(rows, class_filter=args.class_filter, offset=args.offset, limit=args.limit)

    failures = []
    processed = 0
    skipped_existing = 0
    resolved_failures: set[str] = set()
    for row in rows:
        cache_path = _cache_path(row["video_path"])
        if args.skip_existing and cache_path.exists():
            skipped_existing += 1
            continue
        try:
            result = extract_video_landmarks(Path(row["video_path"]))
            payload = {
                "video_path": np.array(result["video_path"]),
                "frame_count": np.array([result["frame_count"]], dtype=np.int32),
                "frame_shape_json": np.array(json.dumps(result["frame_shape"])),
                "feature_spec_json": np.array(json.dumps(result["feature_spec"], ensure_ascii=False)),
            }
            for key, value in result.items():
                if key in {"video_path", "frame_count", "frame_shape", "feature_spec"}:
                    continue
                payload[key] = value
            np.savez_compressed(cache_path, **payload)
            processed += 1
            resolved_failures.add(row["video_path"])
        except Exception as exc:
            failures.append(
                {
                    "video_path": row["video_path"],
                    "english_label": row["english_label"],
                    "class_id": row["class_id"],
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=2).replace("\n", " | "),
                }
            )

    merged_failures = merge_failure_rows(
        existing_failure_rows,
        failures,
        key_field="video_path",
        resolved_keys=resolved_failures,
    )
    write_csv_rows(
        failure_path,
        ["video_path", "english_label", "class_id", "error", "traceback"],
        merged_failures,
    )
    refresh_pipeline_state(
        {
            "script": "extract_landmarks_batch.py",
            "processed_count": processed,
            "failed_count": len(failures),
            "skipped_existing_count": skipped_existing,
            "retry_failed": args.retry_failed,
            "offset": args.offset,
            "limit": args.limit,
            "class_filter": args.class_filter or [],
        }
    )

    print(f"Processed caches: {processed}")
    print(f"Failures this run: {len(failures)}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Failure log: {failure_path}")


if __name__ == "__main__":
    main()
