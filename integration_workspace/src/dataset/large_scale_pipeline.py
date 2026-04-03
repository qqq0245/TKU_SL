from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from config import CONFIG
from src.landmarks.feature_builder import build_frame_feature, get_feature_spec
from src.landmarks.holistic_extractor import HolisticExtractor
from src.utils.paths import ensure_dir


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def slugify_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    normalized = normalized.strip("_")
    return normalized or "unknown_label"


def load_vocabulary_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_label_rows(vocabulary_csv: Path) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for row in load_vocabulary_rows(vocabulary_csv):
        english_label = row["english_label"].strip()
        if english_label in seen:
            continue
        seen.add(english_label)
        rows.append(
            {
                "class_id": len(rows),
                "english_label": english_label,
                "label_slug": slugify_label(english_label),
                "zh_tw_translation": row.get("zh_tw_translation", "").strip(),
                "video_count": int(row.get("video_count", 0) or 0),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_video_source_root(source_root: Path | None = None) -> Path:
    if source_root is not None:
        return source_root
    has_video_structure = False
    if CONFIG.raw_videos_dir.exists():
        for item in CONFIG.raw_videos_dir.iterdir():
            if item.is_dir():
                has_video_structure = True
                break
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                has_video_structure = True
                break
    if has_video_structure:
        return CONFIG.raw_videos_dir
    return CONFIG.external_video_source_dir


def iter_video_files(label_dir: Path):
    for path in sorted(label_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def extract_video_landmarks(video_path: Path) -> dict:
    extractor = HolisticExtractor()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        extractor.close()
        raise RuntimeError(f"Unable to open video: {video_path}")

    raw_left = []
    raw_right = []
    raw_pose = []
    raw_mouth = []
    raw_chin = []
    left_masks = []
    right_masks = []
    pose_masks = []
    mouth_masks = []
    chin_masks = []
    normalized_left = []
    normalized_right = []
    normalized_pose = []
    frame_valid_masks = []
    frame_shape = None

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_shape is None:
                frame_shape = list(frame.shape)
            landmarks, _ = extractor.extract(frame)
            built = build_frame_feature(
                left_hand=landmarks.left_hand,
                right_hand=landmarks.right_hand,
                pose=landmarks.pose,
                mouth_center=landmarks.mouth_center,
                chin=landmarks.chin,
                left_hand_mask=landmarks.left_hand_mask,
                right_hand_mask=landmarks.right_hand_mask,
                pose_mask=landmarks.pose_mask,
                mouth_mask=landmarks.mouth_mask,
                chin_mask=landmarks.chin_mask,
                prev_feature=None,
                feature_mode="landmarks_only",
            )
            raw_left.append(landmarks.left_hand.astype(np.float32))
            raw_right.append(landmarks.right_hand.astype(np.float32))
            raw_pose.append(landmarks.pose.astype(np.float32))
            raw_mouth.append(landmarks.mouth_center.astype(np.float32))
            raw_chin.append(landmarks.chin.astype(np.float32))
            left_masks.append(landmarks.left_hand_mask.astype(np.float32))
            right_masks.append(landmarks.right_hand_mask.astype(np.float32))
            pose_masks.append(landmarks.pose_mask.astype(np.float32))
            mouth_masks.append(landmarks.mouth_mask.astype(np.float32))
            chin_masks.append(landmarks.chin_mask.astype(np.float32))
            normalized_left.append(built["left_hand"].astype(np.float32))
            normalized_right.append(built["right_hand"].astype(np.float32))
            normalized_pose.append(built["pose"].astype(np.float32))
            frame_valid_masks.append(built["frame_valid_mask"].astype(np.float32))
    finally:
        capture.release()
        extractor.close()

    frame_count = len(raw_left)
    if frame_count == 0:
        raise RuntimeError(f"Video has zero readable frames: {video_path}")

    return {
        "video_path": str(video_path),
        "frame_count": frame_count,
        "frame_shape": frame_shape or [],
        "feature_spec": get_feature_spec(CONFIG.feature_mode),
        "raw_left_hand": np.stack(raw_left, axis=0),
        "raw_right_hand": np.stack(raw_right, axis=0),
        "raw_pose": np.stack(raw_pose, axis=0),
        "raw_mouth_center": np.stack(raw_mouth, axis=0),
        "raw_chin": np.stack(raw_chin, axis=0),
        "left_hand_mask": np.stack(left_masks, axis=0),
        "right_hand_mask": np.stack(right_masks, axis=0),
        "pose_mask": np.stack(pose_masks, axis=0),
        "mouth_mask": np.stack(mouth_masks, axis=0),
        "chin_mask": np.stack(chin_masks, axis=0),
        "normalized_left_hand": np.stack(normalized_left, axis=0),
        "normalized_right_hand": np.stack(normalized_right, axis=0),
        "normalized_pose": np.stack(normalized_pose, axis=0),
        "frame_valid_mask": np.stack(frame_valid_masks, axis=0),
    }


def build_feature_sequence_from_cache(cache_data, feature_mode: str) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    frame_valid_masks = []
    prev_feature = None
    frame_count_array = np.asarray(cache_data["frame_count"])
    frame_count = int(frame_count_array.reshape(-1)[0])
    for frame_index in range(frame_count):
        built = build_frame_feature(
            left_hand=cache_data["raw_left_hand"][frame_index],
            right_hand=cache_data["raw_right_hand"][frame_index],
            pose=cache_data["raw_pose"][frame_index],
            mouth_center=cache_data["raw_mouth_center"][frame_index],
            chin=cache_data["raw_chin"][frame_index],
            left_hand_mask=cache_data["left_hand_mask"][frame_index],
            right_hand_mask=cache_data["right_hand_mask"][frame_index],
            pose_mask=cache_data["pose_mask"][frame_index],
            mouth_mask=cache_data["mouth_mask"][frame_index],
            chin_mask=cache_data["chin_mask"][frame_index],
            prev_feature=prev_feature,
            feature_mode=feature_mode,
        )
        prev_feature = built
        sequences.append(built["feature_vector"])
        frame_valid_masks.append(built["frame_valid_mask"])
    return np.stack(sequences, axis=0).astype(np.float32), np.stack(frame_valid_masks, axis=0).astype(np.float32)


def make_fixed_windows(sequence: np.ndarray, frame_valid_mask: np.ndarray, sequence_length: int, stride: int) -> list[dict]:
    frame_count, feature_dim = sequence.shape
    if frame_count <= sequence_length:
        padded_sequence = np.zeros((sequence_length, feature_dim), dtype=np.float32)
        padded_mask = np.zeros((sequence_length, frame_valid_mask.shape[1]), dtype=np.float32)
        padded_sequence[:frame_count] = sequence
        padded_mask[:frame_count] = frame_valid_mask
        return [
            {
                "sequence": padded_sequence,
                "frame_valid_mask": padded_mask,
                "start_frame": 0,
                "end_frame": frame_count - 1,
                "padded_frames": sequence_length - frame_count,
            }
        ]

    windows = []
    for start in range(0, frame_count - sequence_length + 1, stride):
        end = start + sequence_length
        windows.append(
            {
                "sequence": sequence[start:end].astype(np.float32),
                "frame_valid_mask": frame_valid_mask[start:end].astype(np.float32),
                "start_frame": start,
                "end_frame": end - 1,
                "padded_frames": 0,
            }
        )
    if not windows:
        windows.append(
            {
                "sequence": sequence[:sequence_length].astype(np.float32),
                "frame_valid_mask": frame_valid_mask[:sequence_length].astype(np.float32),
                "start_frame": 0,
                "end_frame": min(frame_count, sequence_length) - 1,
                "padded_frames": max(0, sequence_length - frame_count),
            }
        )
    return windows


__all__ = [
    "VIDEO_EXTENSIONS",
    "build_feature_sequence_from_cache",
    "build_label_rows",
    "extract_video_landmarks",
    "iter_video_files",
    "load_vocabulary_rows",
    "make_fixed_windows",
    "resolve_video_source_root",
    "slugify_label",
    "write_csv",
    "write_json",
]
