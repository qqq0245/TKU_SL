from __future__ import annotations

import math
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Create speed-augmented copies of a labeled video dataset.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--speed-factors", nargs="+", type=float, default=[0.8, 1.25])
    parser.add_argument("--copy-originals", action="store_true")
    return parser


def iter_video_files(label_dir: Path):
    for path in sorted(label_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def _target_length(frame_count: int, speed_factor: float) -> int:
    if frame_count <= 1:
        return frame_count
    return max(2, int(round(frame_count / speed_factor)))


def _resample_frames(frames: list, speed_factor: float) -> list:
    if len(frames) <= 1:
        return frames
    target_length = _target_length(len(frames), speed_factor)
    if target_length == len(frames):
        return frames
    last_index = len(frames) - 1
    indices = []
    for output_index in range(target_length):
        source_position = output_index * (last_index / max(target_length - 1, 1))
        indices.append(min(last_index, int(round(source_position))))
    return [frames[index] for index in indices]


def _read_video_frames(video_path: Path) -> tuple[list, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Video has no frames: {video_path}")
    if width <= 0 or height <= 0:
        height, width = frames[0].shape[:2]
    return frames, float(fps), (width, height)


def _write_video(video_path: Path, frames: list, fps: float, size: tuple[int, int]) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video: {video_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def _speed_suffix(speed_factor: float) -> str:
    return f"speed{int(round(speed_factor * 100)):03d}"


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    labels = sorted(path for path in source_root.iterdir() if path.is_dir())
    copied_originals = 0
    augmented_videos = 0
    skipped_files = 0

    for label_dir in labels:
        output_label_dir = output_root / label_dir.name
        output_label_dir.mkdir(parents=True, exist_ok=True)
        for video_path in iter_video_files(label_dir):
            if args.copy_originals:
                target_original = output_label_dir / video_path.name
                if not target_original.exists():
                    shutil.copy2(video_path, target_original)
                    copied_originals += 1

            try:
                frames, fps, size = _read_video_frames(video_path)
                for speed_factor in args.speed_factors:
                    speed_frames = _resample_frames(frames, speed_factor)
                    augmented_name = f"{video_path.stem}_{_speed_suffix(speed_factor)}{video_path.suffix.lower()}"
                    augmented_path = output_label_dir / augmented_name
                    _write_video(augmented_path, speed_frames, fps, size)
                    augmented_videos += 1
            except Exception:
                skipped_files += 1

    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print(f"Copied originals: {copied_originals}")
    print(f"Augmented videos: {augmented_videos}")
    print(f"Skipped files: {skipped_files}")


if __name__ == "__main__":
    main()
