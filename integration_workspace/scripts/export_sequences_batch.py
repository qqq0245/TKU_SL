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
from src.dataset.dataset_writer import DatasetWriter
from src.dataset.large_scale_pipeline import build_feature_sequence_from_cache, make_fixed_windows
from src.dataset.pipeline_state import refresh_pipeline_state
from src.utils.paths import ensure_dir


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Export cached landmarks into training-ready sequence npz files.")
    parser.add_argument("--manifest-csv", default=str(CONFIG.manifests_dir / "video_manifest.csv"))
    parser.add_argument("--sequence-length", type=int, default=CONFIG.sequence_length)
    parser.add_argument("--stride", type=int, default=CONFIG.sequence_length)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--class-filter", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def _load_existing_sample_ids(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    sample_ids = set()
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sample_ids.add(json.loads(line)["sample_id"])
    return sample_ids


def _cache_path(video_path: str) -> Path:
    video = Path(video_path)
    return CONFIG.landmarks_cache_dir / video.parent.name / f"{video.stem}.npz"


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(CONFIG.processed_sequences_dir)
    with Path(args.manifest_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    failure_path = CONFIG.dataset_logs_dir / "sequences_failed.csv"
    existing_failure_rows = read_csv_rows(failure_path)
    rows = manifest_rows
    if args.retry_failed:
        failed_paths = {row["source_video_path"] for row in existing_failure_rows}
        rows = [row for row in manifest_rows if row["video_path"] in failed_paths]
    rows = filter_rows(rows, class_filter=args.class_filter, offset=args.offset, limit=args.limit)

    writer = DatasetWriter(CONFIG.processed_sequences_dir)
    existing_sample_ids = _load_existing_sample_ids(writer.index_path) if args.skip_existing else set()
    exported_rows = []
    skipped_existing = 0
    failures = []
    resolved_failures: set[str] = set()

    for row in rows:
        cache_path = _cache_path(row["video_path"])
        if not cache_path.exists():
            failures.append(
                {
                    "source_video_path": row["video_path"],
                    "english_label": row["english_label"],
                    "class_id": row["class_id"],
                    "error": "missing_landmarks_cache",
                    "traceback": "",
                }
            )
            continue
        try:
            cache_data = np.load(cache_path, allow_pickle=True)
            sequence, frame_valid_mask = build_feature_sequence_from_cache(cache_data, CONFIG.feature_mode)
            windows = make_fixed_windows(sequence, frame_valid_mask, args.sequence_length, args.stride)

            for window_index, window in enumerate(windows):
                video_stem = Path(row["video_path"]).stem
                sample_id = f"{row['english_label']}_{video_stem}_{window_index:03d}"
                if sample_id in existing_sample_ids:
                    skipped_existing += 1
                    continue
                metadata = {
                    "label": row["english_label"],
                    "class_id": int(row["class_id"]),
                    "sample_id": sample_id,
                    "source_video_path": row["video_path"],
                    "source_origin_path": row.get("source_origin_path", ""),
                    "source_group_key": row.get("source_group_key", ""),
                    "sequence_length": args.sequence_length,
                    "feature_dim": int(window["sequence"].shape[1]),
                    "feature_mode": CONFIG.feature_mode,
                    "zh_tw_translation": row["zh_tw_translation"],
                    "start_frame": int(window["start_frame"]),
                    "end_frame": int(window["end_frame"]),
                    "padded_frames": int(window["padded_frames"]),
                }
                writer.write_sample(
                    sample_id=sample_id,
                    class_label=row["english_label"],
                    class_id=int(row["class_id"]),
                    sequence=window["sequence"],
                    metadata=metadata,
                    frame_valid_mask=window["frame_valid_mask"],
                )
                exported_rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_path": str((CONFIG.processed_sequences_dir / f"{sample_id}.npz").resolve()),
                        "class_id": int(row["class_id"]),
                        "english_label": row["english_label"],
                        "zh_tw_translation": row["zh_tw_translation"],
                        "feature_mode": CONFIG.feature_mode,
                        "feature_dim": int(window["sequence"].shape[1]),
                        "sequence_length": args.sequence_length,
                        "source_video_path": row["video_path"],
                        "source_origin_path": row.get("source_origin_path", ""),
                        "source_group_key": row.get("source_group_key", ""),
                    }
                )
            resolved_failures.add(row["video_path"])
        except Exception as exc:
            failures.append(
                {
                    "source_video_path": row["video_path"],
                    "english_label": row["english_label"],
                    "class_id": row["class_id"],
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=2).replace("\n", " | "),
                }
            )

    export_manifest = CONFIG.manifests_dir / "sequence_manifest.csv"
    with export_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "sample_path",
                "class_id",
                "english_label",
                "zh_tw_translation",
                "feature_mode",
                "feature_dim",
                "sequence_length",
                "source_video_path",
                "source_origin_path",
                "source_group_key",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(exported_rows)

    merged_failures = merge_failure_rows(
        existing_failure_rows,
        failures,
        key_field="source_video_path",
        resolved_keys=resolved_failures,
    )
    write_csv_rows(
        failure_path,
        ["source_video_path", "english_label", "class_id", "error", "traceback"],
        merged_failures,
    )
    refresh_pipeline_state(
        {
            "script": "export_sequences_batch.py",
            "processed_count": len(exported_rows),
            "failed_count": len(failures),
            "skipped_existing_count": skipped_existing,
            "retry_failed": args.retry_failed,
            "offset": args.offset,
            "limit": args.limit,
            "class_filter": args.class_filter or [],
        }
    )

    print(f"Exported sequences: {len(exported_rows)}")
    print(f"Failures this run: {len(failures)}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Sequence manifest: {export_manifest}")


if __name__ == "__main__":
    main()
