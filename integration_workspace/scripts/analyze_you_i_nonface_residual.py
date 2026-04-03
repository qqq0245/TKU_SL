from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.continuous_feature_cache import load_continuous_feature_cache


EPS = 1e-8
POSE_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]
LOCATION_VECTOR_NAMES = [
    "left_to_nose",
    "left_to_chin",
    "left_to_mouth",
    "left_to_shoulder_center",
    "left_to_chest_center",
    "left_to_torso_center",
    "right_to_nose",
    "right_to_chin",
    "right_to_mouth",
    "right_to_shoulder_center",
    "right_to_chest_center",
    "right_to_torso_center",
    "left_to_right",
]
MOTION_COLUMN_NAMES = [
    "left_velocity_x",
    "left_velocity_y",
    "left_velocity_z",
    "right_velocity_x",
    "right_velocity_y",
    "right_velocity_z",
    "left_speed",
    "right_speed",
    "distance_delta",
    "left_acc_x",
    "left_acc_y",
    "left_acc_z",
    "right_acc_x",
    "right_acc_y",
    "right_acc_z",
    "left_torso_motion_x",
    "left_torso_motion_y",
    "left_torso_motion_z",
    "right_torso_motion_x",
    "right_torso_motion_y",
    "right_torso_motion_z",
    "left_direction_x",
    "left_direction_y",
    "left_direction_z",
    "right_direction_x",
    "right_direction_y",
    "right_direction_z",
    "sync",
]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Analyze the non-face residual that pulls you -> i.")
    parser.add_argument("--spans-json", required=True)
    parser.add_argument("--classification-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--sequence-manifest",
        default=str(ROOT / "dataset_pipeline_webcam9_relative_coord_v1" / "manifests" / "sequence_manifest.csv"),
    )
    parser.add_argument(
        "--processed-sequences-dir",
        default=str(ROOT / "dataset_pipeline_webcam9_relative_coord_v1" / "processed_sequences"),
    )
    parser.add_argument("--mirror-input", type=int, default=1)
    parser.add_argument("--prototype-groups-per-class", type=int, default=6)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: float) -> float:
    return round(float(value), 6)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= EPS:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def _group_slices(feature_spec: dict) -> tuple[dict[str, list[int]], dict[str, dict[str, object]]]:
    channels = int(CONFIG.channels)
    left_nodes = int(CONFIG.left_hand_nodes)
    right_nodes = int(CONFIG.right_hand_nodes)
    pose_nodes = len(CONFIG.pose_indices)
    left_xyz = left_nodes * channels
    right_xyz = right_nodes * channels
    pose_coord_start = left_xyz + right_xyz
    pose_xyz = pose_nodes * channels
    left_mask_start = pose_coord_start + pose_xyz
    right_mask_start = left_mask_start + left_nodes
    pose_mask_start = right_mask_start + right_nodes

    explicit_range = feature_spec["landmark_sections"]["explicit_finger_states"]
    explicit_start = int(explicit_range["start"])
    explicit_end = int(explicit_range["end"])

    location_range = feature_spec["components"]["location"]
    location_start = int(location_range["start"])
    location_end = int(location_range["end"])
    left_face_start = location_start
    left_body_start = left_face_start + 9
    right_face_start = left_body_start + 9
    right_body_start = right_face_start + 9
    left_right_start = right_body_start + 9
    left_zone_start = left_right_start + 3
    right_zone_start = left_zone_start + len(CONFIG.location_zone_names)
    location_validity_start = location_end - 4

    motion_range = feature_spec["components"]["motion"]
    motion_start = int(motion_range["start"])
    motion_end = int(motion_range["end"])
    motion_validity_start = motion_end - 2

    pose_torso_nodes = [0, 1, 2, 7, 8]
    pose_arm_nodes = [3, 4, 5, 6]

    def pose_coord_indices(nodes: list[int]) -> list[int]:
        indices: list[int] = []
        for node_index in nodes:
            base = pose_coord_start + node_index * channels
            indices.extend(range(base, base + channels))
        return indices

    def pose_mask_indices(nodes: list[int]) -> list[int]:
        return [pose_mask_start + node_index for node_index in nodes]

    groups = {
        "hand_local_graph": list(range(0, left_xyz + right_xyz)),
        "left_hand_local_graph": list(range(0, left_xyz)),
        "right_hand_local_graph": list(range(left_xyz, left_xyz + right_xyz)),
        "pose_context_graph": list(range(pose_coord_start, pose_coord_start + pose_xyz))
        + list(range(pose_mask_start, pose_mask_start + pose_nodes)),
        "pose_torso_frame": pose_coord_indices(pose_torso_nodes) + pose_mask_indices(pose_torso_nodes),
        "pose_arm_chain": pose_coord_indices(pose_arm_nodes) + pose_mask_indices(pose_arm_nodes),
        "location_vectors_nonface": list(range(left_body_start, left_body_start + 9))
        + list(range(right_body_start, right_body_start + 9))
        + list(range(left_right_start, left_right_start + 3))
        + list(range(left_zone_start, left_zone_start + len(CONFIG.location_zone_names)))
        + list(range(right_zone_start, right_zone_start + len(CONFIG.location_zone_names))),
        "left_body_vectors": list(range(left_body_start, left_body_start + 9)),
        "right_body_vectors": list(range(right_body_start, right_body_start + 9)),
        "left_right_vector": list(range(left_right_start, left_right_start + 3)),
        "left_zone": list(range(left_zone_start, left_zone_start + len(CONFIG.location_zone_names))),
        "right_zone": list(range(right_zone_start, right_zone_start + len(CONFIG.location_zone_names))),
        "left_pose_location_joint": list(range(left_body_start, left_body_start + 9))
        + list(range(left_zone_start, left_zone_start + len(CONFIG.location_zone_names)))
        + pose_coord_indices(pose_arm_nodes)
        + pose_mask_indices(pose_arm_nodes),
        "explicit_finger_state": list(range(explicit_start, explicit_end)),
        "left_finger_state": list(range(explicit_start, explicit_start + 5)),
        "right_finger_state": list(range(explicit_start + 5, explicit_end)),
        "motion_core": list(range(motion_start, motion_validity_start)),
        "left_motion_core": list(range(motion_start, motion_start + 16)),
        "right_motion_core": list(range(motion_start + 3, motion_start + 21)),
    }
    metadata = {
        "pose_coord_start": pose_coord_start,
        "pose_mask_start": pose_mask_start,
        "left_body_start": left_body_start,
        "right_body_start": right_body_start,
        "left_right_start": left_right_start,
        "left_zone_start": left_zone_start,
        "right_zone_start": right_zone_start,
        "motion_start": motion_start,
        "location_validity_start": location_validity_start,
        "motion_validity_start": motion_validity_start,
        "pose_torso_nodes": pose_torso_nodes,
        "pose_arm_nodes": pose_arm_nodes,
    }
    return groups, metadata


def _sequence_summary(sequence: np.ndarray, indices: list[int]) -> dict[str, object]:
    subset = sequence[:, indices].astype(np.float32)
    mean_vector = subset.mean(axis=0)
    std_vector = subset.std(axis=0)
    frame_energy = np.linalg.norm(subset, axis=1)
    peak_index = int(np.argmax(frame_energy)) if len(frame_energy) else 0
    peak_value = float(frame_energy[peak_index]) if len(frame_energy) else 0.0
    return {
        "mean_vector": mean_vector,
        "std_vector": std_vector,
        "feature_mean": _safe_float(np.mean(mean_vector)),
        "feature_std": _safe_float(np.mean(std_vector)),
        "feature_energy": _safe_float(np.mean(frame_energy ** 2)),
        "zero_ratio": _safe_float(np.mean(np.abs(subset) <= 1e-6)),
        "frame_activation_mean": _safe_float(np.mean(frame_energy)),
        "frame_activation_std": _safe_float(np.std(frame_energy)),
        "peak_frame_offset": peak_index,
        "peak_frame_energy": _safe_float(peak_value),
        "frame_energy_curve": [_safe_float(value) for value in frame_energy.tolist()],
    }


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
    channels = int(CONFIG.channels)
    left_xyz = int(CONFIG.left_hand_nodes) * channels
    right_xyz = int(CONFIG.right_hand_nodes) * channels
    pose_coord_start = left_xyz + right_xyz
    pose_coord_end = pose_coord_start + (len(CONFIG.pose_indices) * channels)
    for row_index, frame_index in enumerate(sampled_indices.astype(int).tolist()):
        adjusted[row_index, pose_coord_start:pose_coord_end] = arrays["normalized_pose"][frame_index].reshape(-1).astype(
            np.float32
        )
    return adjusted


def _build_class_prototypes(
    *,
    manifest_rows: list[dict[str, str]],
    processed_sequences_dir: Path,
    class_labels: set[str],
    group_indices: dict[str, list[int]],
    groups_per_class: int,
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
    chosen_paths: dict[str, list[Path]] = defaultdict(list)
    seen_groups: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        label = str(row["english_label"]).strip().lower()
        if label not in class_labels:
            continue
        if len(chosen_paths[label]) >= groups_per_class:
            continue
        group_key = str(row.get("source_group_key", "")).strip()
        if group_key in seen_groups[label]:
            continue
        sample_id = str(row["sample_id"]).strip()
        sample_path = processed_sequences_dir / f"{sample_id}.npz"
        if not sample_path.exists():
            continue
        seen_groups[label].add(group_key)
        chosen_paths[label].append(sample_path)

    prototypes: dict[str, dict[str, object]] = {}
    chosen_serialized: dict[str, list[str]] = {}
    for label, paths in chosen_paths.items():
        chosen_serialized[label] = [str(path) for path in paths]
        group_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        group_std_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        scalar_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                sequence = payload["sequence"].astype(np.float32)
            for group_name, indices in group_indices.items():
                summary = _sequence_summary(sequence, indices)
                group_vectors[group_name].append(summary["mean_vector"])
                group_std_vectors[group_name].append(summary["std_vector"])
                for key in ("feature_mean", "feature_std", "feature_energy", "zero_ratio", "frame_activation_mean", "frame_activation_std", "peak_frame_energy"):
                    scalar_stats[group_name][key].append(float(summary[key]))
        prototypes[label] = {"sample_count": len(paths), "groups": {}}
        for group_name in group_indices:
            if not group_vectors[group_name]:
                continue
            prototypes[label]["groups"][group_name] = {
                "prototype_mean_vector": np.stack(group_vectors[group_name], axis=0).mean(axis=0),
                "prototype_std_vector": np.stack(group_std_vectors[group_name], axis=0).mean(axis=0),
                "feature_mean": _safe_float(np.mean(scalar_stats[group_name]["feature_mean"])),
                "feature_std": _safe_float(np.mean(scalar_stats[group_name]["feature_std"])),
                "feature_energy": _safe_float(np.mean(scalar_stats[group_name]["feature_energy"])),
                "zero_ratio": _safe_float(np.mean(scalar_stats[group_name]["zero_ratio"])),
                "frame_activation_mean": _safe_float(np.mean(scalar_stats[group_name]["frame_activation_mean"])),
                "frame_activation_std": _safe_float(np.mean(scalar_stats[group_name]["frame_activation_std"])),
                "peak_frame_energy": _safe_float(np.mean(scalar_stats[group_name]["peak_frame_energy"])),
            }
    return prototypes, chosen_serialized


def _distance_summary(span_summary: dict[str, object], prototype_summary: dict[str, object]) -> dict[str, float]:
    span_mean = span_summary["mean_vector"]
    prototype_mean = prototype_summary["prototype_mean_vector"]
    span_std = span_summary["std_vector"]
    prototype_std = prototype_summary["prototype_std_vector"]
    return {
        "mean_vector_l2": _safe_float(np.linalg.norm(span_mean - prototype_mean)),
        "mean_vector_cosine": _safe_float(_cosine_distance(span_mean, prototype_mean)),
        "std_vector_l2": _safe_float(np.linalg.norm(span_std - prototype_std)),
        "energy_gap": _safe_float(abs(float(span_summary["feature_energy"]) - float(prototype_summary["feature_energy"]))),
        "zero_ratio_gap": _safe_float(abs(float(span_summary["zero_ratio"]) - float(prototype_summary["zero_ratio"]))),
    }


def _takeover_offset(sequence: np.ndarray, indices: list[int], true_proto: np.ndarray, confused_proto: np.ndarray) -> dict[str, object]:
    subset = sequence[:, indices].astype(np.float32)
    prefix_means = np.cumsum(subset, axis=0) / np.arange(1, subset.shape[0] + 1, dtype=np.float32)[:, None]
    deltas: list[float] = []
    first_positive = -1
    for offset, prefix_mean in enumerate(prefix_means):
        delta = float(np.linalg.norm(prefix_mean - true_proto) - np.linalg.norm(prefix_mean - confused_proto))
        deltas.append(_safe_float(delta))
        if delta > 0 and first_positive < 0:
            first_positive = offset
    return {
        "first_true_to_confused_takeover_offset": int(first_positive),
        "prefix_true_minus_confused_l2": deltas,
    }


def _column_breakdown(
    *,
    group_name: str,
    sequence: np.ndarray,
    indices: list[int],
    true_proto: np.ndarray,
    confused_proto: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, object]]:
    subset = sequence[:, indices].astype(np.float32)
    mean_vector = subset.mean(axis=0)
    rows: list[dict[str, object]] = []
    for offset, name in enumerate(feature_names):
        span_value = float(mean_vector[offset])
        true_value = float(true_proto[offset])
        confused_value = float(confused_proto[offset])
        true_gap = abs(span_value - true_value)
        confused_gap = abs(span_value - confused_value)
        rows.append(
            {
                "name": name,
                "span_mean": _safe_float(span_value),
                "true_prototype_mean": _safe_float(true_value),
                "confused_prototype_mean": _safe_float(confused_value),
                "span_minus_true": _safe_float(span_value - true_value),
                "span_minus_confused": _safe_float(span_value - confused_value),
                "true_minus_confused_gap": _safe_float(true_gap - confused_gap),
            }
        )
    rows.sort(key=lambda item: abs(float(item["true_minus_confused_gap"])), reverse=True)
    return rows


def _pose_feature_names(nodes: list[int]) -> list[str]:
    names: list[str] = []
    for node in nodes:
        node_name = POSE_NAMES[node]
        names.extend([f"{node_name}_x", f"{node_name}_y", f"{node_name}_z"])
    for node in nodes:
        names.append(f"{POSE_NAMES[node]}_mask")
    return names


def _group_feature_names(group_name: str, metadata: dict[str, object]) -> list[str]:
    if group_name == "left_body_vectors":
        return [f"{name}_{axis}" for name in LOCATION_VECTOR_NAMES[3:6] for axis in ("x", "y", "z")]
    if group_name == "right_body_vectors":
        return [f"{name}_{axis}" for name in LOCATION_VECTOR_NAMES[9:12] for axis in ("x", "y", "z")]
    if group_name == "left_right_vector":
        return [f"{LOCATION_VECTOR_NAMES[12]}_{axis}" for axis in ("x", "y", "z")]
    if group_name == "left_zone":
        return [f"left_zone_{name}" for name in CONFIG.location_zone_names]
    if group_name == "right_zone":
        return [f"right_zone_{name}" for name in CONFIG.location_zone_names]
    if group_name == "pose_torso_frame":
        return _pose_feature_names(list(metadata["pose_torso_nodes"]))
    if group_name == "pose_arm_chain":
        return _pose_feature_names(list(metadata["pose_arm_nodes"]))
    if group_name == "explicit_finger_state":
        return list(CONFIG.enable_explicit_finger_states and (
            [
                "left_thumb",
                "left_index",
                "left_middle",
                "left_ring",
                "left_pinky",
                "right_thumb",
                "right_index",
                "right_middle",
                "right_ring",
                "right_pinky",
            ]
        ) or [])
    if group_name == "left_finger_state":
        return ["left_thumb", "left_index", "left_middle", "left_ring", "left_pinky"]
    if group_name == "right_finger_state":
        return ["right_thumb", "right_index", "right_middle", "right_ring", "right_pinky"]
    if group_name == "motion_core":
        return list(MOTION_COLUMN_NAMES)
    if group_name == "left_motion_core":
        return MOTION_COLUMN_NAMES[:16]
    if group_name == "right_motion_core":
        return MOTION_COLUMN_NAMES[3:21]
    return [f"{group_name}_{index}" for index in range(0)]


def _inference_from_ranked_groups(ranked_groups: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    if not ranked_groups:
        return "", {}
    best = ranked_groups[0]
    group_name = str(best["group_name"])
    if group_name == "left_body_vectors":
        return (
            "left_body_vectors",
            {
                "dominant_semantics": "left-hand body-relative pointing semantics against shoulder/chest/torso references",
                "minimal_fix_direction": "pairwise you/i left-body-vector calibration",
            },
        )
    if group_name == "left_pose_location_joint":
        return (
            "left_pose_location_joint",
            {
                "dominant_semantics": "left-hand body-relative pointing coupled with pose-arm context",
                "minimal_fix_direction": "pairwise you/i left pose-location joint calibration",
            },
        )
    if group_name == "pose_torso_frame":
        return (
            "pose_torso_frame",
            {
                "dominant_semantics": "torso-frame pointing / body-relative pose normalization",
                "minimal_fix_direction": "pairwise you/i torso-frame calibration",
            },
        )
    return (
        group_name,
        {
            "dominant_semantics": "non-face residual concentrated in a single scorer subgroup",
            "minimal_fix_direction": f"pairwise you/i {group_name} calibration",
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    spans_path = Path(args.spans_json).resolve()
    classification_path = Path(args.classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()
    manifest_path = Path(args.sequence_manifest).resolve()
    processed_sequences_dir = Path(args.processed_sequences_dir).resolve()
    mirror_input = int(args.mirror_input)

    spans_payload = _load_json(spans_path)
    classification_payload = _load_json(classification_path)
    cache_path = cache_dir / f"continuous_feature_cache_mirror{mirror_input}.npz"
    metadata, arrays = load_continuous_feature_cache(cache_path)
    feature_vectors = arrays["feature_vectors"].astype(np.float32)
    feature_spec = metadata["feature_spec"]
    group_indices, group_metadata = _group_slices(feature_spec)
    manifest_rows = _read_csv_rows(manifest_path)
    prototypes, prototype_paths = _build_class_prototypes(
        manifest_rows=manifest_rows,
        processed_sequences_dir=processed_sequences_dir,
        class_labels={"you", "i"},
        group_indices=group_indices,
        groups_per_class=int(args.prototype_groups_per_class),
    )

    token_predictions = {
        str(row["token"]).strip().lower(): row
        for row in classification_payload.get("classifications", [])
        if int(row.get("mirror_input", -1)) == mirror_input
    }
    you_span = next(
        span for span in spans_payload.get("token_span_hypothesis", [])
        if str(span.get("token", "")).strip().lower() == "you"
    )
    classification = token_predictions["you"]
    sampled_indices = np.asarray(classification.get("sampled_frame_indices"), dtype=np.int32)
    sequence = _rebuild_pose_local_sequence(
        base_sequence=feature_vectors[sampled_indices],
        arrays=arrays,
        sampled_indices=sampled_indices,
        pose_local_anchor=str(classification_payload.get("pose_local_anchor", "")),
    )

    ranked_groups: list[dict[str, object]] = []
    subgroup_payloads: dict[str, dict[str, object]] = {}
    for group_name, indices in group_indices.items():
        span_summary = _sequence_summary(sequence, indices)
        true_proto = prototypes["you"]["groups"][group_name]
        confused_proto = prototypes["i"]["groups"][group_name]
        true_dist = _distance_summary(span_summary, true_proto)
        confused_dist = _distance_summary(span_summary, confused_proto)
        mean_gap = float(true_dist["mean_vector_l2"]) - float(confused_dist["mean_vector_l2"])
        takeover = _takeover_offset(
            sequence,
            indices,
            true_proto["prototype_mean_vector"],
            confused_proto["prototype_mean_vector"],
        )
        subgroup_payload = {
            "feature_mean": span_summary["feature_mean"],
            "feature_std": span_summary["feature_std"],
            "feature_energy": span_summary["feature_energy"],
            "zero_ratio": span_summary["zero_ratio"],
            "frame_activation_mean": span_summary["frame_activation_mean"],
            "frame_activation_std": span_summary["frame_activation_std"],
            "peak_frame_offset": int(span_summary["peak_frame_offset"]),
            "peak_frame_energy": span_summary["peak_frame_energy"],
            "distance_to_true_you": true_dist,
            "distance_to_confused_i": confused_dist,
            "true_minus_confused_l2": _safe_float(mean_gap),
            "takeover": takeover,
        }
        subgroup_payloads[group_name] = subgroup_payload
        ranked_groups.append(
            {
                "group_name": group_name,
                "true_minus_confused_l2": _safe_float(mean_gap),
                "first_true_to_confused_takeover_offset": int(takeover["first_true_to_confused_takeover_offset"]),
                "feature_energy": span_summary["feature_energy"],
                "frame_activation_mean": span_summary["frame_activation_mean"],
            }
        )

    ranked_groups.sort(
        key=lambda item: (
            float(item["true_minus_confused_l2"]),
            -1.0 * float(item["feature_energy"]),
            -1.0 * float(item["frame_activation_mean"]),
        ),
        reverse=True,
    )
    dominant_subgroup, recommended_fix = _inference_from_ranked_groups(ranked_groups)
    dominant_feature_names = _group_feature_names(dominant_subgroup, group_metadata)
    dominant_indices = group_indices[dominant_subgroup]
    dominant_breakdown = _column_breakdown(
        group_name=dominant_subgroup,
        sequence=sequence,
        indices=dominant_indices,
        true_proto=prototypes["you"]["groups"][dominant_subgroup]["prototype_mean_vector"],
        confused_proto=prototypes["i"]["groups"][dominant_subgroup]["prototype_mean_vector"],
        feature_names=dominant_feature_names if dominant_feature_names else [f"{dominant_subgroup}_{i}" for i in range(len(dominant_indices))],
    )

    output_payload = {
        "spans_json": str(spans_path),
        "classification_json": str(classification_path),
        "cache_path": str(cache_path),
        "mirror_input": mirror_input,
        "pose_local_anchor": str(classification_payload.get("pose_local_anchor", "")),
        "target_token": "you",
        "confused_token": "i",
        "baseline_prediction": str(classification.get("predicted_label", "")),
        "baseline_top3": classification.get("top3", []),
        "sampled_frame_indices": sampled_indices.astype(int).tolist(),
        "prototype_groups_per_class": int(args.prototype_groups_per_class),
        "prototype_paths": prototype_paths,
        "subgroup_rankings": ranked_groups,
        "dominant_nonface_subgroup": dominant_subgroup,
        "subgroups": subgroup_payloads,
        "dominant_subgroup_column_breakdown": dominant_breakdown,
        "recommended_minimal_fix": recommended_fix,
        "analysis_inference": (
            f"you->i is dominated by {dominant_subgroup}, "
            f"where the you span is closer to i than to you under non-face-only prototype distance."
        ),
        "root_cause_cut": {
            "dominant_nonface_subgroup": dominant_subgroup,
            "dominant_semantics": recommended_fix.get("dominant_semantics", ""),
            "minimal_fix_direction": recommended_fix.get("minimal_fix_direction", ""),
        },
        "you_span": {
            "start_frame": int(you_span["start_frame"]),
            "end_frame": int(you_span["end_frame"]),
            "active_frame_count": int(you_span["active_frame_count"]),
        },
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
