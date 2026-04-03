from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.landmarks.feature_builder import build_frame_feature, resolve_feature_spec
from src.landmarks.holistic_extractor import HolisticExtractor


def build_continuous_feature_cache(
    video_path: Path,
    *,
    mirror_input: bool,
    feature_mode: str,
    feature_spec: dict | None,
    capture_backend: int | None = None,
) -> dict[str, object]:
    backend = capture_backend if capture_backend is not None else getattr(cv2, "CAP_FFMPEG", 0)
    capture = cv2.VideoCapture(str(video_path), backend)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    extractor = HolisticExtractor()
    property_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

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
    origin_rows = []
    scale_rows = []
    normalized_left = []
    normalized_right = []
    normalized_pose = []
    normalized_mouth = []
    normalized_chin = []
    normalized_left_center = []
    normalized_right_center = []
    normalized_nose = []
    normalized_shoulder_center = []
    normalized_chest_center = []
    normalized_torso_center = []
    landmark_features = []
    location_features = []
    motion_features = []
    feature_vectors = []
    explicit_finger_state_features = []
    frame_valid_masks = []
    frame_indices = []
    prev_feature = None
    frame_shape = None
    frame_count = 0

    spec = resolve_feature_spec(feature_mode=feature_mode, feature_spec=feature_spec)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_shape is None:
                frame_shape = list(frame.shape)
            current = cv2.flip(frame, 1) if mirror_input else frame
            landmarks, _ = extractor.extract(current)
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
                prev_feature=prev_feature,
                feature_mode=feature_mode,
                feature_spec=spec,
            )
            prev_feature = built

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
            origin_rows.append(built["origin"].astype(np.float32))
            scale_rows.append(built["scale"].astype(np.float32))
            normalized_left.append(built["left_hand"].astype(np.float32))
            normalized_right.append(built["right_hand"].astype(np.float32))
            normalized_pose.append(built["pose"].astype(np.float32))
            normalized_mouth.append(built["mouth_center"].astype(np.float32))
            normalized_chin.append(built["chin"].astype(np.float32))
            normalized_left_center.append(built["left_hand_center"].astype(np.float32))
            normalized_right_center.append(built["right_hand_center"].astype(np.float32))
            refs = built["reference_points"]
            normalized_nose.append(refs["nose"].astype(np.float32))
            normalized_shoulder_center.append(refs["shoulder_center"].astype(np.float32))
            normalized_chest_center.append(refs["chest_center"].astype(np.float32))
            normalized_torso_center.append(refs["torso_center"].astype(np.float32))
            landmark_features.append(built["landmark_features"].astype(np.float32))
            location_features.append(built["location_features"].astype(np.float32))
            motion_features.append(built["motion_features"].astype(np.float32))
            feature_vectors.append(built["feature_vector"].astype(np.float32))
            explicit_finger_state_features.append(built["explicit_finger_state_features"].astype(np.float32))
            frame_valid_masks.append(built["frame_valid_mask"].astype(np.float32))
            frame_indices.append(frame_count)
            frame_count += 1
    finally:
        capture.release()
        extractor.close()

    if frame_count == 0:
        raise RuntimeError(f"Video has zero readable frames: {video_path}")

    arrays = {
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
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
        "origin": np.stack(origin_rows, axis=0),
        "scale": np.stack(scale_rows, axis=0),
        "normalized_left_hand": np.stack(normalized_left, axis=0),
        "normalized_right_hand": np.stack(normalized_right, axis=0),
        "normalized_pose": np.stack(normalized_pose, axis=0),
        "normalized_mouth_center": np.stack(normalized_mouth, axis=0),
        "normalized_chin": np.stack(normalized_chin, axis=0),
        "normalized_left_hand_center": np.stack(normalized_left_center, axis=0),
        "normalized_right_hand_center": np.stack(normalized_right_center, axis=0),
        "normalized_nose": np.stack(normalized_nose, axis=0),
        "normalized_shoulder_center": np.stack(normalized_shoulder_center, axis=0),
        "normalized_chest_center": np.stack(normalized_chest_center, axis=0),
        "normalized_torso_center": np.stack(normalized_torso_center, axis=0),
        "explicit_finger_state_features": np.stack(explicit_finger_state_features, axis=0),
        "landmark_features": np.stack(landmark_features, axis=0),
        "location_features": np.stack(location_features, axis=0),
        "motion_features": np.stack(motion_features, axis=0),
        "feature_vectors": np.stack(feature_vectors, axis=0),
        "frame_valid_mask": np.stack(frame_valid_masks, axis=0),
    }
    metadata = {
        "cache_version": 1,
        "video_path": str(video_path),
        "mirror_input": int(mirror_input),
        "capture_backend": int(backend),
        "property_frame_count": property_frame_count,
        "readable_frame_count": frame_count,
        "frame_shape": frame_shape or [],
        "feature_mode": feature_mode,
        "feature_spec": spec,
        "array_names": sorted(arrays.keys()),
    }
    return {"metadata": metadata, "arrays": arrays}


def save_continuous_feature_cache(cache_payload: dict[str, object], cache_path: Path) -> tuple[Path, Path]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(cache_payload["metadata"])
    arrays = dict(cache_payload["arrays"])
    np.savez_compressed(cache_path, **arrays)
    metadata_path = cache_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_path, metadata_path


def load_continuous_feature_cache(cache_path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata_path = cache_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing cache metadata JSON: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return metadata, arrays
