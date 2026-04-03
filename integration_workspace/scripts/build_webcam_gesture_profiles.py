from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from argparse import ArgumentParser


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.landmarks.feature_builder import build_frame_feature


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build gesture range profiles from webcam landmarks cache.")
    parser.add_argument(
        "--manifest-csv",
        default=str(
            PROJECT_ROOT
            / "integration_workspace"
            / "dataset_pipeline_webcam9_nosign_seq30s5"
            / "manifests"
            / "video_manifest.csv"
        ),
    )
    parser.add_argument(
        "--cache-root",
        default=str(
            PROJECT_ROOT
            / "integration_workspace"
            / "dataset_pipeline_webcam9_nosign_seq30s5"
            / "landmarks_cache"
        ),
    )
    parser.add_argument(
        "--output-json",
        default=str(PROJECT_ROOT / "metadata" / "webcam9_gesture_profiles.json"),
    )
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--orig-only", action="store_true", default=True)
    return parser


def _dominant_hand(frame_feature: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    if bool(frame_feature["right_hand_valid"][0] > 0):
        return (
            frame_feature["right_hand"],
            frame_feature["right_hand_mask"],
            frame_feature["right_hand_center"],
            True,
        )
    return (
        frame_feature["left_hand"],
        frame_feature["left_hand_mask"],
        frame_feature["left_hand_center"],
        bool(frame_feature["left_hand_valid"][0] > 0),
    )


def _finger_extended(hand: np.ndarray, hand_mask: np.ndarray, tip_idx: int, pip_idx: int) -> bool:
    if hand_mask[tip_idx] <= 0 or hand_mask[pip_idx] <= 0 or hand_mask[0] <= 0:
        return False
    tip_to_wrist = float(np.linalg.norm(hand[tip_idx] - hand[0]))
    pip_to_wrist = float(np.linalg.norm(hand[pip_idx] - hand[0]))
    return tip_to_wrist > pip_to_wrist + 0.03


def _cache_path(cache_root: Path, video_path: str) -> Path:
    video = Path(video_path)
    return cache_root / video.parent.name / f"{video.stem}.npz"


def _iter_feature_rows(cache_data) -> list[dict[str, float]]:
    frame_count = int(np.asarray(cache_data["frame_count"]).reshape(-1)[0])
    prev_feature = None
    feature_rows: list[dict[str, float]] = []
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
            feature_mode="landmarks_plus_location_motion",
        )
        prev_feature = built
        hand, hand_mask, hand_center, hand_valid = _dominant_hand(built)
        if not hand_valid or built["frame_valid_mask"][2] <= 0:
            continue

        refs = built["reference_points"]
        index_extended = _finger_extended(hand, hand_mask, 8, 6)
        middle_extended = _finger_extended(hand, hand_mask, 12, 10)
        ring_extended = _finger_extended(hand, hand_mask, 16, 14)
        pinky_extended = _finger_extended(hand, hand_mask, 20, 18)
        extended_count = sum(int(flag) for flag in (index_extended, middle_extended, ring_extended, pinky_extended))
        open_palm = extended_count >= 3
        index_only = index_extended and not pinky_extended and extended_count <= 2
        pinky_only = pinky_extended and not index_extended and extended_count <= 2
        chin_valid = bool(built["chin_mask"][0] > 0)
        mouth_valid = bool(built["mouth_mask"][0] > 0)
        thumb_tip = hand[4] if hand_mask[4] > 0 else hand_center

        feature_rows.append(
            {
                "center_x": float(hand_center[0]),
                "center_y": float(hand_center[1]),
                "chin_distance": float(np.linalg.norm(hand_center - refs["chin"])) if chin_valid else 999.0,
                "mouth_distance": float(np.linalg.norm(hand_center - refs["mouth_center"])) if mouth_valid else 999.0,
                "chest_distance": float(np.linalg.norm(hand_center - refs["chest_center"])),
                "thumb_chin_distance": float(np.linalg.norm(thumb_tip - refs["chin"])) if chin_valid else 999.0,
                "index_only": float(index_only),
                "pinky_only": float(pinky_only),
                "open_palm": float(open_palm),
                "extended_count": float(extended_count),
            }
        )
    return feature_rows


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    args = build_parser().parse_args()
    manifest_csv = Path(args.manifest_csv)
    cache_root = Path(args.cache_root)
    output_json = Path(args.output_json)
    allowed_video_paths: set[str] | None = None
    if args.split_csv:
        with Path(args.split_csv).open("r", encoding="utf-8-sig", newline="") as handle:
            allowed_video_paths = {
                str(Path(row["source_video_path"]).resolve())
                for row in csv.DictReader(handle)
                if row.get("source_video_path")
            }

    grouped_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            video_path = row["video_path"]
            if allowed_video_paths is not None and str(Path(video_path).resolve()) not in allowed_video_paths:
                continue
            if args.orig_only and "_orig." not in video_path.lower():
                continue
            cache_path = _cache_path(cache_root, video_path)
            if not cache_path.exists():
                continue
            cache_data = np.load(cache_path, allow_pickle=True)
            grouped_rows[row["english_label"]].extend(_iter_feature_rows(cache_data))

    profiles = {}
    for label, rows in sorted(grouped_rows.items()):
        if not rows:
            continue
        feature_names = rows[0].keys()
        profiles[label] = {
            "frame_count": len(rows),
            "features": {
                name: _summary([float(row[name]) for row in rows])
                for name in feature_names
            },
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Labels profiled: {len(profiles)}")
    print(f"Output: {output_json}")


if __name__ == "__main__":
    main()
