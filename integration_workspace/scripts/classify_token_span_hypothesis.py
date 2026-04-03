from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.continuous_feature_cache import load_continuous_feature_cache
from src.landmarks.location_features import build_location_features
from src.models.feature_slices import split_feature_tensor
from src.models.feature_group_transforms import apply_hand_mask_validity_scale, apply_pose_hip_coordinate_scale
from src.models.inference_utils_multibranch import load_multibranch_checkpoint


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Classify token-span hypotheses using a full-video continuous feature cache.")
    parser.add_argument("--spans-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "artifacts_webcam9_relative_coord_v1" / "models" / "multibranch_baseline.pt"),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--mirror-modes",
        default="1",
        help="Comma-separated mirror_input values to score from the cache dir, e.g. 1 or 0,1.",
    )
    parser.add_argument("--hand-mask-validity-scale", type=float, default=1.0)
    parser.add_argument("--pose-hip-coordinate-scale", type=float, default=1.0)
    parser.add_argument("--left-location-anchor", default="center")
    parser.add_argument("--right-location-anchor", default="center")
    parser.add_argument(
        "--left-face-reference-modes",
        default="nose,chin,mouth_center",
        help="Comma-separated face reference names for left face vectors (nose/chin/mouth_center).",
    )
    parser.add_argument(
        "--left-face-anchor-modes",
        default="center,center,center",
        help="Comma-separated left face anchor modes for nose/chin/mouth vectors.",
    )
    parser.add_argument("--pose-local-anchor", default="mid_shoulder")
    return parser


def _parse_mirror_modes(raw_value: str) -> list[bool]:
    values: list[bool] = []
    seen: set[int] = set()
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value not in {0, 1}:
            raise ValueError(f"Unsupported mirror mode: {token}")
        if value in seen:
            continue
        seen.add(value)
        values.append(bool(value))
    if not values:
        raise ValueError("At least one mirror mode is required")
    return values


def _parse_triplet(raw_value: str, *, valid_values: set[str], arg_name: str) -> tuple[str, str, str]:
    values = tuple(str(token).strip().lower() for token in raw_value.split(",") if str(token).strip())
    if len(values) != 3:
        raise ValueError(f"{arg_name} must contain exactly 3 comma-separated values: {raw_value!r}")
    invalid = [value for value in values if value not in valid_values]
    if invalid:
        raise ValueError(f"Unsupported {arg_name} value(s): {invalid!r}")
    return values


def _rebuild_location_sequence(
    *,
    base_sequence: np.ndarray,
    arrays: dict[str, np.ndarray],
    sampled_indices: np.ndarray,
    feature_spec: dict,
    left_anchor_mode: str,
    right_anchor_mode: str,
    left_face_reference_modes: tuple[str, str, str],
    left_face_anchor_modes: tuple[str, str, str],
) -> np.ndarray:
    if (
        left_anchor_mode == "center"
        and right_anchor_mode == "center"
        and left_face_reference_modes == ("nose", "chin", "mouth_center")
        and left_face_anchor_modes == ("center", "center", "center")
    ):
        return base_sequence
    adjusted = base_sequence.copy()
    location_start = int(feature_spec["components"]["location"]["start"])
    location_end = int(feature_spec["components"]["location"]["end"])
    for row_index, frame_index in enumerate(sampled_indices.astype(int).tolist()):
        left_mask = arrays["left_hand_mask"][frame_index].astype(np.float32)
        right_mask = arrays["right_hand_mask"][frame_index].astype(np.float32)
        pose_mask = arrays["pose_mask"][frame_index].astype(np.float32)
        mouth_mask = arrays["mouth_mask"][frame_index].astype(np.float32)
        chin_mask = arrays["chin_mask"][frame_index].astype(np.float32)
        frame_feature = {
            "left_hand": arrays["normalized_left_hand"][frame_index].astype(np.float32),
            "right_hand": arrays["normalized_right_hand"][frame_index].astype(np.float32),
            "left_hand_mask": left_mask,
            "right_hand_mask": right_mask,
            "pose_mask": pose_mask,
            "mouth_mask": mouth_mask,
            "chin_mask": chin_mask,
            "left_hand_center": arrays["normalized_left_hand_center"][frame_index].astype(np.float32),
            "right_hand_center": arrays["normalized_right_hand_center"][frame_index].astype(np.float32),
            "left_hand_valid": np.array([float(np.any(left_mask > 0))], dtype=np.float32),
            "right_hand_valid": np.array([float(np.any(right_mask > 0))], dtype=np.float32),
            "reference_points": {
                "nose": arrays["normalized_nose"][frame_index].astype(np.float32),
                "shoulder_center": arrays["normalized_shoulder_center"][frame_index].astype(np.float32),
                "chest_center": arrays["normalized_chest_center"][frame_index].astype(np.float32),
                "torso_center": arrays["normalized_torso_center"][frame_index].astype(np.float32),
                "mouth_center": arrays["normalized_mouth_center"][frame_index].astype(np.float32),
                "chin": arrays["normalized_chin"][frame_index].astype(np.float32),
            },
        }
        location = build_location_features(
            frame_feature,
            left_anchor_mode=left_anchor_mode,
            right_anchor_mode=right_anchor_mode,
            left_face_reference_modes=left_face_reference_modes,
            left_face_anchor_modes=left_face_anchor_modes,
        )["feature_vector"].astype(np.float32)
        adjusted[row_index, location_start:location_end] = location
    return adjusted


def _rebuild_pose_local_sequence(
    *,
    base_sequence: np.ndarray,
    arrays: dict[str, np.ndarray],
    sampled_indices: np.ndarray,
    pose_local_anchor: str,
) -> np.ndarray:
    anchor_mode = str(pose_local_anchor).strip().lower()
    if anchor_mode == "mid_shoulder":
        return base_sequence
    if anchor_mode != "torso_center":
        raise ValueError(f"Unsupported pose-local anchor: {pose_local_anchor}")
    adjusted = base_sequence.copy()
    channels = 3
    left_xyz = 21 * channels
    right_xyz = 21 * channels
    pose_coord_start = left_xyz + right_xyz
    pose_coord_end = pose_coord_start + (9 * channels)
    for row_index, frame_index in enumerate(sampled_indices.astype(int).tolist()):
        adjusted[row_index, pose_coord_start:pose_coord_end] = arrays["normalized_pose"][frame_index].reshape(-1).astype(np.float32)
    return adjusted


@torch.no_grad()
def _predict_top3(
    model,
    *,
    feature_mode: str,
    feature_spec: dict | None,
    index_to_label: dict[int, str],
    sequence: np.ndarray,
    hand_mask_validity_scale: float,
    pose_hip_coordinate_scale: float,
) -> tuple[list[dict[str, float | str]], np.ndarray]:
    adjusted_sequence = apply_pose_hip_coordinate_scale(
        sequence,
        feature_spec=feature_spec,
        scale=pose_hip_coordinate_scale,
    )
    adjusted_sequence = apply_hand_mask_validity_scale(
        adjusted_sequence,
        feature_spec=feature_spec,
        scale=hand_mask_validity_scale,
    )
    tensor = torch.from_numpy(adjusted_sequence).unsqueeze(0)
    parts = split_feature_tensor(tensor, feature_mode, feature_spec=feature_spec)
    logits = model(parts["skeleton_stream"], parts["location_stream"], parts["motion_stream"])
    logits_np = logits.squeeze(0).detach().cpu().numpy()
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    top_indices = np.argsort(-probabilities)[:3]
    top3 = []
    for index in top_indices:
        index_value = int(index)
        top3.append(
            {
                "label": index_to_label[index_value],
                "confidence": round(float(probabilities[index_value]), 6),
                "logit": round(float(logits_np[index_value]), 6),
            }
        )
    return top3, logits_np


def main() -> None:
    args = build_parser().parse_args()
    spans_path = Path(args.spans_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_json = Path(args.output_json).resolve()

    payload = json.loads(spans_path.read_text(encoding="utf-8"))
    spans = payload.get("token_span_hypothesis", [])
    device = torch.device("cpu")
    model, checkpoint = load_multibranch_checkpoint(checkpoint_path, device)
    if model is None or checkpoint is None:
        raise RuntimeError(f"Unable to load checkpoint: {checkpoint_path}")
    model.eval()
    feature_mode = str(checkpoint["feature_mode"])
    feature_spec = checkpoint.get("feature_spec")
    index_to_label = checkpoint["index_to_label"]
    mirror_modes = _parse_mirror_modes(args.mirror_modes)
    left_face_reference_modes = _parse_triplet(
        str(args.left_face_reference_modes),
        valid_values={"nose", "chin", "mouth_center"},
        arg_name="left-face-reference-modes",
    )
    left_face_anchor_modes = _parse_triplet(
        str(args.left_face_anchor_modes),
        valid_values={"center", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip", "closest_valid_fingertip"},
        arg_name="left-face-anchor-modes",
    )

    classifications: list[dict[str, object]] = []
    summary_by_mirror: list[dict[str, object]] = []
    for mirror_input in mirror_modes:
        cache_path = cache_dir / f"continuous_feature_cache_mirror{int(mirror_input)}.npz"
        metadata, arrays = load_continuous_feature_cache(cache_path)
        cache = arrays["feature_vectors"]
        frame_count = int(metadata["readable_frame_count"])
        predicted_counts: Counter[str] = Counter()
        confusion: dict[str, Counter[str]] = defaultdict(Counter)
        reference_tokens: list[str] = []
        predicted_tokens: list[str] = []
        correct_count = 0
        for span in spans:
            start_frame = int(span["start_frame"])
            end_frame = min(int(span["end_frame"]), frame_count - 1)
            if end_frame < start_frame:
                continue
            sampled_indices = (
                np.linspace(start_frame, end_frame, 30).round().astype(int)
                if end_frame > start_frame
                else np.full((30,), start_frame, dtype=int)
            )
            sequence = cache[sampled_indices].astype(np.float32)
            sequence = _rebuild_location_sequence(
                base_sequence=sequence,
                arrays=arrays,
                sampled_indices=sampled_indices,
                feature_spec=feature_spec,
                left_anchor_mode=str(args.left_location_anchor).strip().lower(),
                right_anchor_mode=str(args.right_location_anchor).strip().lower(),
                left_face_reference_modes=left_face_reference_modes,
                left_face_anchor_modes=left_face_anchor_modes,
            )
            sequence = _rebuild_pose_local_sequence(
                base_sequence=sequence,
                arrays=arrays,
                sampled_indices=sampled_indices,
                pose_local_anchor=str(args.pose_local_anchor).strip().lower(),
            )
            top3, logits_np = _predict_top3(
                model,
                feature_mode=feature_mode,
                feature_spec=feature_spec,
                index_to_label=index_to_label,
                sequence=sequence,
                hand_mask_validity_scale=float(args.hand_mask_validity_scale),
                pose_hip_coordinate_scale=float(args.pose_hip_coordinate_scale),
            )
            predicted_label = top3[0]["label"] if top3 else ""
            correct = int(bool(top3 and predicted_label == span["token"]))
            reference_tokens.append(str(span["token"]))
            predicted_tokens.append(predicted_label)
            predicted_counts[predicted_label] += 1
            confusion[str(span["token"])][predicted_label] += 1
            correct_count += correct
            classifications.append(
                {
                    "mirror_input": int(mirror_input),
                    "token": span["token"],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "active_frame_count": int(end_frame - start_frame + 1),
                    "sampled_frame_indices": sampled_indices.astype(int).tolist(),
                    "top3": top3,
                    "predicted_label": predicted_label,
                    "correct": correct,
                    "top1_logit": round(float(logits_np[np.argmax(logits_np)]), 6),
                }
            )
        total_count = len(reference_tokens)
        summary_by_mirror.append(
            {
                "mirror_input": int(mirror_input),
                "frame_count": frame_count,
                "token_count": total_count,
                "correct_count": correct_count,
                "exact_span_accuracy": round((correct_count / total_count) if total_count else 0.0, 6),
                "reference_tokens": reference_tokens,
                "predicted_tokens": predicted_tokens,
                "predicted_label_counts": dict(predicted_counts),
                "confusion": {label: dict(counts) for label, counts in confusion.items()},
            }
        )

    output_payload = {
        "spans_json": str(spans_path),
        "cache_dir": str(cache_dir),
        "checkpoint_path": str(checkpoint_path),
        "feature_mode": feature_mode,
        "feature_spec": feature_spec,
        "hand_mask_validity_scale": float(args.hand_mask_validity_scale),
        "pose_hip_coordinate_scale": float(args.pose_hip_coordinate_scale),
        "left_location_anchor": str(args.left_location_anchor).strip().lower(),
        "right_location_anchor": str(args.right_location_anchor).strip().lower(),
        "left_face_reference_modes": list(left_face_reference_modes),
        "left_face_anchor_modes": list(left_face_anchor_modes),
        "pose_local_anchor": str(args.pose_local_anchor).strip().lower(),
        "summary_by_mirror": summary_by_mirror,
        "classifications": classifications,
    }
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_json}")
    for row in classifications:
        print(
            f"mirror={row['mirror_input']} {row['token']} {row['start_frame']}-{row['end_frame']} "
            f"-> {row['predicted_label']}"
        )


if __name__ == "__main__":
    main()
