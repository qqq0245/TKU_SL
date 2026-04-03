from __future__ import annotations

import csv
import random
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import ensure_dir


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build a webcam word dataset and synthesize no_sign clips from idle head/tail segments.")
    parser.add_argument(
        "--source-root",
        default=str(PROJECT_ROOT / "datasets" / "recorded" / "webcam_30_words"),
    )
    parser.add_argument(
        "--labels-csv",
        default=str(PROJECT_ROOT / "metadata" / "webcam_8_vocabulary.csv"),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "datasets" / "training_variants" / "webcam_9_with_nosign_raw"),
    )
    parser.add_argument(
        "--explicit-nosign-dir",
        default="",
        help="Optional directory of manually recorded no_sign clips to copy into output_root/no_sign.",
    )
    parser.add_argument("--target-nosign-clips", type=int, default=64)
    parser.add_argument("--segment-frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_labels(labels_csv: Path) -> list[str]:
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return [str(row["english_label"]).strip().lower() for row in csv.DictReader(handle)]


def load_source_metadata(source_root: Path) -> dict[str, dict]:
    metadata_path = source_root / "source_metadata.csv"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(Path(row["video_path"]).resolve()): row for row in rows if row.get("video_path")}


def iter_video_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def copy_word_videos(source_root: Path, output_root: Path, labels: list[str], source_metadata: dict[str, dict]) -> list[dict]:
    copied_rows: list[dict] = []
    for label in labels:
        src_dir = source_root / label
        dst_dir = ensure_dir(output_root / label)
        for input_video in iter_video_files(src_dir):
            output_video = dst_dir / input_video.name
            shutil.copy2(input_video, output_video)
            metadata = source_metadata.get(str(input_video.resolve()), {})
            copied_rows.append(
                {
                    "video_path": str(output_video.resolve()),
                    "source_origin_path": metadata.get("source_origin_path", str(input_video.resolve())),
                    "source_group_key": metadata.get("source_group_key", str(input_video.resolve())),
                }
            )
    return copied_rows


def copy_explicit_nosign_videos(explicit_nosign_dir: Path, output_root: Path) -> tuple[list[dict], list[dict]]:
    if not explicit_nosign_dir.exists():
        return [], []
    dst_dir = ensure_dir(output_root / "no_sign")
    metadata_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for index, input_video in enumerate(iter_video_files(explicit_nosign_dir), start=1):
        output_video = dst_dir / input_video.name
        shutil.copy2(input_video, output_video)
        metadata_rows.append(
            {
                "video_path": str(output_video.resolve()),
                "source_origin_path": str(input_video.resolve()),
                "source_group_key": str(input_video.resolve()),
            }
        )
        manifest_rows.append(
            {
                "clip_id": f"recorded_no_sign_{index:03d}",
                "source_label": "no_sign",
                "source_video_path": str(input_video.resolve()),
                "segment_type": "recorded",
                "frame_count": "",
                "motion_score": "",
                "output_video_path": str(output_video.resolve()),
                "clip_type": "recorded",
            }
        )
    return metadata_rows, manifest_rows


def read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames, fps


def compute_motion_score(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    diffs = []
    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diffs.append(float(np.mean(cv2.absdiff(current, previous))))
        previous = current
    return float(np.mean(diffs)) if diffs else 0.0


def write_video(frames: list[np.ndarray], output_video: Path, fps: float) -> None:
    if not frames:
        raise ValueError(f"No frames to write: {output_video}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to write video: {output_video}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def collect_nosign_candidates(source_root: Path, labels: list[str], segment_frames: int) -> list[dict]:
    candidates: list[dict] = []
    for label in labels:
        for video_path in iter_video_files(source_root / label):
            frames, fps = read_video_frames(video_path)
            if len(frames) < max(4, segment_frames):
                continue
            segments = {
                "head": frames[:segment_frames],
                "tail": frames[-segment_frames:],
            }
            for segment_type, segment_frames_data in segments.items():
                candidates.append(
                    {
                        "source_label": label,
                        "source_video_path": str(video_path.resolve()),
                        "segment_type": segment_type,
                        "fps": fps,
                        "frame_count": len(segment_frames_data),
                        "motion_score": compute_motion_score(segment_frames_data),
                        "frames": segment_frames_data,
                    }
                )
    return candidates


def select_nosign_candidates(candidates: list[dict], target_nosign_clips: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: item["motion_score"])
    selected: list[dict] = []
    used_sources: set[str] = set()
    for candidate in candidates:
        source_video_path = candidate["source_video_path"]
        if source_video_path in used_sources:
            continue
        selected.append(candidate)
        used_sources.add(source_video_path)
        if len(selected) >= target_nosign_clips:
            break
    return selected


def write_nosign_clips(output_root: Path, candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    nosign_dir = ensure_dir(output_root / "no_sign")
    manifest_rows: list[dict] = []
    metadata_rows: list[dict] = []
    for index, candidate in enumerate(candidates, start=1):
        source_video = Path(candidate["source_video_path"])
        output_video = nosign_dir / f"no_sign_{index:03d}_{source_video.stem}_{candidate['segment_type']}.mp4"
        write_video(candidate["frames"], output_video, candidate["fps"])
        metadata_rows.append(
            {
                "video_path": str(output_video.resolve()),
                "source_origin_path": candidate["source_video_path"],
                "source_group_key": candidate["source_video_path"],
            }
        )
        manifest_rows.append(
            {
                "clip_id": f"no_sign_{index:03d}",
                "source_label": candidate["source_label"],
                "source_video_path": candidate["source_video_path"],
                "segment_type": candidate["segment_type"],
                "frame_count": candidate["frame_count"],
                "motion_score": f"{candidate['motion_score']:.6f}",
                "output_video_path": str(output_video.resolve()),
                "clip_type": "synthetic",
            }
        )
    return manifest_rows, metadata_rows


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root)
    labels_csv = Path(args.labels_csv)
    output_root = Path(args.output_root)
    explicit_nosign_dir = Path(args.explicit_nosign_dir) if args.explicit_nosign_dir else (source_root / "no_sign")
    labels = load_labels(labels_csv)
    source_metadata = load_source_metadata(source_root)

    copied_metadata_rows = copy_word_videos(source_root, output_root, labels, source_metadata)
    explicit_nosign_metadata_rows, explicit_nosign_manifest_rows = copy_explicit_nosign_videos(
        explicit_nosign_dir,
        output_root,
    )
    candidates = collect_nosign_candidates(source_root, labels, args.segment_frames)
    selected = select_nosign_candidates(candidates, args.target_nosign_clips, args.seed)
    manifest_rows, nosign_metadata_rows = write_nosign_clips(output_root, selected)
    combined_nosign_manifest_rows = explicit_nosign_manifest_rows + manifest_rows

    manifest_path = output_root / "no_sign_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_id",
                "source_label",
                "source_video_path",
                "segment_type",
                "frame_count",
                "motion_score",
                "output_video_path",
                "clip_type",
            ],
        )
        writer.writeheader()
        writer.writerows(combined_nosign_manifest_rows)

    source_metadata_path = output_root / "source_metadata.csv"
    with source_metadata_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_path", "source_origin_path", "source_group_key"],
        )
        writer.writeheader()
        writer.writerows(copied_metadata_rows + explicit_nosign_metadata_rows + nosign_metadata_rows)

    print(f"Labels copied: {len(labels)}")
    print(f"Recorded no_sign clips copied: {len(explicit_nosign_manifest_rows)}")
    print(f"No-sign candidates: {len(candidates)}")
    print(f"No-sign clips written: {len(manifest_rows)}")
    print(f"Output root: {output_root}")
    print(f"No-sign manifest: {manifest_path}")
    print(f"Source metadata: {source_metadata_path}")


if __name__ == "__main__":
    main()
