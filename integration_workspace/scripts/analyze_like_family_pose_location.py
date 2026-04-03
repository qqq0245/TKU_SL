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
from src.landmarks.location_features import LOCATION_VECTOR_NAMES


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


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Analyze like-family pose/location drift against true and like prototypes.")
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: float) -> float:
    return round(float(value), 6)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= EPS:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def _group_slices(feature_spec: dict) -> dict[str, list[int]]:
    channels = int(CONFIG.channels)
    left_xyz = int(CONFIG.left_hand_nodes) * channels
    right_xyz = int(CONFIG.right_hand_nodes) * channels
    pose_nodes = len(CONFIG.pose_indices)
    pose_xyz = pose_nodes * channels
    left_mask_start = left_xyz + right_xyz + pose_xyz
    right_mask_start = left_mask_start + int(CONFIG.left_hand_nodes)
    pose_mask_start = right_mask_start + int(CONFIG.right_hand_nodes)
    location_range = feature_spec["components"]["location"]
    location_start = int(location_range["start"])

    pose_coord_start = left_xyz + right_xyz
    left_face_start = location_start
    left_body_start = left_face_start + 9
    right_face_start = left_body_start + 9
    right_body_start = right_face_start + 9
    left_right_start = right_body_start + 9
    left_zone_start = left_right_start + 3
    right_zone_start = left_zone_start + len(CONFIG.location_zone_names)

    pose_torso_indices = [0, 1, 2, 7, 8]
    pose_arm_indices = [3, 4, 5, 6]

    def pose_coord_indices(nodes: list[int]) -> list[int]:
        indices: list[int] = []
        for node_index in nodes:
            base = pose_coord_start + node_index * channels
            indices.extend(range(base, base + channels))
        return indices

    def pose_mask_indices(nodes: list[int]) -> list[int]:
        return [pose_mask_start + node_index for node_index in nodes]

    return {
        "left_face_vectors": list(range(left_face_start, left_face_start + 9)),
        "left_body_vectors": list(range(left_body_start, left_body_start + 9)),
        "left_zone": list(range(left_zone_start, left_zone_start + len(CONFIG.location_zone_names))),
        "pose_torso_frame": pose_coord_indices(pose_torso_indices) + pose_mask_indices(pose_torso_indices),
        "pose_arm_chain": pose_coord_indices(pose_arm_indices) + pose_mask_indices(pose_arm_indices),
        "left_pose_location_joint": list(range(left_body_start, left_body_start + 9))
        + list(range(left_zone_start, left_zone_start + len(CONFIG.location_zone_names)))
        + pose_coord_indices(pose_arm_indices)
        + pose_mask_indices(pose_arm_indices),
    }


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
        label = row["english_label"].strip().lower()
        if label not in class_labels:
            continue
        if len(chosen_paths[label]) >= groups_per_class:
            continue
        group_key = row.get("source_group_key", "")
        if group_key in seen_groups[label]:
            continue
        sample_id = row["sample_id"].strip()
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
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                sequence = payload["sequence"].astype(np.float32)
            for group_name, indices in group_indices.items():
                summary = _sequence_summary(sequence, indices)
                group_vectors[group_name].append(summary["mean_vector"])
                group_std_vectors[group_name].append(summary["std_vector"])
        prototypes[label] = {"sample_count": len(paths), "groups": {}}
        for group_name in group_indices:
            if not group_vectors[group_name]:
                continue
            prototypes[label]["groups"][group_name] = {
                "prototype_mean_vector": np.stack(group_vectors[group_name], axis=0).mean(axis=0),
                "prototype_std_vector": np.stack(group_std_vectors[group_name], axis=0).mean(axis=0),
            }
    return prototypes, chosen_serialized


def _distance(span_mean: np.ndarray, proto_mean: np.ndarray) -> dict[str, float]:
    return {
        "mean_vector_l2": _safe_float(np.linalg.norm(span_mean - proto_mean)),
        "mean_vector_cosine": _safe_float(_cosine_distance(span_mean, proto_mean)),
    }


def _takeover_offset(sequence: np.ndarray, indices: list[int], reference_proto: np.ndarray, like_proto: np.ndarray) -> dict[str, object]:
    subset = sequence[:, indices].astype(np.float32)
    prefix_means = np.cumsum(subset, axis=0) / np.arange(1, subset.shape[0] + 1, dtype=np.float32)[:, None]
    deltas = []
    first_positive = -1
    for offset, prefix_mean in enumerate(prefix_means):
        delta = float(np.linalg.norm(prefix_mean - reference_proto) - np.linalg.norm(prefix_mean - like_proto))
        deltas.append(_safe_float(delta))
        if delta > 0 and first_positive < 0:
            first_positive = offset
    return {
        "first_reference_to_like_takeover_offset": first_positive,
        "prefix_reference_minus_like_l2": deltas,
    }


def _vector_component_breakdown(
    *,
    sequence: np.ndarray,
    start: int,
    names: list[str],
    prototype_reference: np.ndarray,
    prototype_like: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, name in enumerate(names):
        col_start = start + offset * 3
        col_end = col_start + 3
        subset = sequence[:, col_start:col_end].astype(np.float32)
        mean_vector = subset.mean(axis=0)
        ref_proto = prototype_reference[offset * 3 : offset * 3 + 3]
        like_proto = prototype_like[offset * 3 : offset * 3 + 3]
        ref_l2 = float(np.linalg.norm(mean_vector - ref_proto))
        like_l2 = float(np.linalg.norm(mean_vector - like_proto))
        rows.append(
            {
                "name": name,
                "mean_vector": [_safe_float(value) for value in mean_vector.tolist()],
                "reference_l2": _safe_float(ref_l2),
                "like_l2": _safe_float(like_l2),
                "reference_minus_like_l2": _safe_float(ref_l2 - like_l2),
            }
        )
    rows.sort(key=lambda item: item["reference_minus_like_l2"], reverse=True)
    return rows


def _zone_component_breakdown(
    *,
    sequence: np.ndarray,
    start: int,
    names: list[str],
    prototype_reference: np.ndarray,
    prototype_like: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, name in enumerate(names):
        column = start + offset
        values = sequence[:, column].astype(np.float32)
        mean_value = float(values.mean())
        ref_mean = float(prototype_reference[offset])
        like_mean = float(prototype_like[offset])
        ref_gap = abs(mean_value - ref_mean)
        like_gap = abs(mean_value - like_mean)
        rows.append(
            {
                "name": name,
                "mean_activation": _safe_float(mean_value),
                "reference_mean": _safe_float(ref_mean),
                "like_mean": _safe_float(like_mean),
                "reference_gap": _safe_float(ref_gap),
                "like_gap": _safe_float(like_gap),
                "reference_minus_like_gap": _safe_float(ref_gap - like_gap),
            }
        )
    rows.sort(key=lambda item: item["reference_minus_like_gap"], reverse=True)
    return rows


def _pose_component_breakdown(
    *,
    sequence: np.ndarray,
    pose_coord_start: int,
    pose_mask_start: int,
    pose_names: list[str],
    selected_nodes: list[int],
    prototype_reference: np.ndarray,
    prototype_like: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cursor = 0
    for node_index in selected_nodes:
        coord_start = pose_coord_start + node_index * 3
        coord_end = coord_start + 3
        coord_values = sequence[:, coord_start:coord_end].astype(np.float32)
        mask_values = sequence[:, pose_mask_start + node_index].astype(np.float32)
        mean_vector = coord_values.mean(axis=0)
        ref_proto = prototype_reference[cursor : cursor + 3]
        like_proto = prototype_like[cursor : cursor + 3]
        ref_mask = float(prototype_reference[len(selected_nodes) * 3 + selected_nodes.index(node_index)])
        like_mask = float(prototype_like[len(selected_nodes) * 3 + selected_nodes.index(node_index)])
        ref_l2 = float(np.linalg.norm(mean_vector - ref_proto))
        like_l2 = float(np.linalg.norm(mean_vector - like_proto))
        rows.append(
            {
                "name": pose_names[node_index],
                "mean_vector": [_safe_float(value) for value in mean_vector.tolist()],
                "mask_mean": _safe_float(mask_values.mean()),
                "reference_l2": _safe_float(ref_l2),
                "like_l2": _safe_float(like_l2),
                "reference_mask_gap": _safe_float(abs(float(mask_values.mean()) - ref_mask)),
                "like_mask_gap": _safe_float(abs(float(mask_values.mean()) - like_mask)),
                "reference_minus_like_l2": _safe_float(ref_l2 - like_l2),
            }
        )
        cursor += 3
    rows.sort(key=lambda item: item["reference_minus_like_l2"], reverse=True)
    return rows


def main() -> None:
    args = build_parser().parse_args()
    spans_path = Path(args.spans_json).resolve()
    classification_path = Path(args.classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()
    manifest_path = Path(args.sequence_manifest).resolve()
    processed_sequences_dir = Path(args.processed_sequences_dir).resolve()
    mirror_input = bool(args.mirror_input)

    spans_payload = _load_json(spans_path)
    classification_payload = _load_json(classification_path)
    cache_path = cache_dir / f"continuous_feature_cache_mirror{int(mirror_input)}.npz"
    metadata, arrays = load_continuous_feature_cache(cache_path)
    feature_vectors = arrays["feature_vectors"].astype(np.float32)
    feature_spec = metadata["feature_spec"]
    group_indices = _group_slices(feature_spec)
    manifest_rows = _read_csv_rows(manifest_path)
    class_labels = {"i", "you", "mother", "like"}
    prototypes, prototype_paths = _build_class_prototypes(
        manifest_rows=manifest_rows,
        processed_sequences_dir=processed_sequences_dir,
        class_labels=class_labels,
        group_indices=group_indices,
        groups_per_class=int(args.prototype_groups_per_class),
    )

    location_start = int(feature_spec["components"]["location"]["start"])
    pose_coord_start = int(CONFIG.left_hand_nodes + CONFIG.right_hand_nodes) * int(CONFIG.channels)
    pose_mask_start = int(feature_spec["landmark_sections"]["graph"]["end"]) - len(CONFIG.pose_indices)
    left_zone_start = location_start + 39
    token_predictions = {
        str(row["token"]).strip().lower(): row
        for row in classification_payload.get("classifications", [])
        if int(row.get("mirror_input", -1)) == int(mirror_input)
    }

    fine_group_deltas: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, object]] = []
    for span in spans_payload.get("token_span_hypothesis", []):
        token = str(span["token"]).strip().lower()
        if token not in {"i", "you", "mother"}:
            continue
        classification = token_predictions.get(token, {})
        start_frame = int(span["start_frame"])
        end_frame = int(span["end_frame"])
        sampled_indices = np.asarray(classification.get("sampled_frame_indices"), dtype=np.int32)
        if sampled_indices.size == 0:
            sampled_indices = (
                np.linspace(start_frame, end_frame, 30).round().astype(np.int32)
                if end_frame > start_frame
                else np.full((30,), start_frame, dtype=np.int32)
            )
        sequence = feature_vectors[sampled_indices]
        token_row = {
            "token": token,
            "predicted_label": str(classification.get("predicted_label", "")),
            "sampled_frame_indices": sampled_indices.astype(int).tolist(),
            "subgroups": {},
        }
        for group_name, indices in group_indices.items():
            summary = _sequence_summary(sequence, indices)
            ref_proto = prototypes[token]["groups"][group_name]["prototype_mean_vector"]
            like_proto = prototypes["like"]["groups"][group_name]["prototype_mean_vector"]
            ref_dist = _distance(summary["mean_vector"], ref_proto)
            like_dist = _distance(summary["mean_vector"], like_proto)
            delta = float(ref_dist["mean_vector_l2"]) - float(like_dist["mean_vector_l2"])
            fine_group_deltas[group_name].append(delta)
            subgroup_payload = {
                "feature_mean": summary["feature_mean"],
                "feature_std": summary["feature_std"],
                "feature_energy": summary["feature_energy"],
                "zero_ratio": summary["zero_ratio"],
                "frame_activation_mean": summary["frame_activation_mean"],
                "frame_activation_std": summary["frame_activation_std"],
                "peak_frame_offset": int(summary["peak_frame_offset"]),
                "peak_frame_energy": summary["peak_frame_energy"],
                "distance_to_reference": ref_dist,
                "distance_to_like": like_dist,
                "reference_minus_like_l2": _safe_float(delta),
                "takeover": _takeover_offset(sequence, indices, ref_proto, like_proto),
            }
            token_row["subgroups"][group_name] = subgroup_payload

        left_face_ref = prototypes[token]["groups"]["left_face_vectors"]["prototype_mean_vector"]
        left_face_like = prototypes["like"]["groups"]["left_face_vectors"]["prototype_mean_vector"]
        left_body_ref = prototypes[token]["groups"]["left_body_vectors"]["prototype_mean_vector"]
        left_body_like = prototypes["like"]["groups"]["left_body_vectors"]["prototype_mean_vector"]
        left_zone_ref = prototypes[token]["groups"]["left_zone"]["prototype_mean_vector"]
        left_zone_like = prototypes["like"]["groups"]["left_zone"]["prototype_mean_vector"]
        pose_torso_ref = prototypes[token]["groups"]["pose_torso_frame"]["prototype_mean_vector"]
        pose_torso_like = prototypes["like"]["groups"]["pose_torso_frame"]["prototype_mean_vector"]
        pose_arm_ref = prototypes[token]["groups"]["pose_arm_chain"]["prototype_mean_vector"]
        pose_arm_like = prototypes["like"]["groups"]["pose_arm_chain"]["prototype_mean_vector"]

        token_row["component_breakdown"] = {
            "left_face_vectors": _vector_component_breakdown(
                sequence=sequence,
                start=location_start,
                names=LOCATION_VECTOR_NAMES[:3],
                prototype_reference=left_face_ref,
                prototype_like=left_face_like,
            ),
            "left_body_vectors": _vector_component_breakdown(
                sequence=sequence,
                start=location_start + 9,
                names=LOCATION_VECTOR_NAMES[3:6],
                prototype_reference=left_body_ref,
                prototype_like=left_body_like,
            ),
            "left_zone": _zone_component_breakdown(
                sequence=sequence,
                start=left_zone_start,
                names=list(CONFIG.location_zone_names),
                prototype_reference=left_zone_ref,
                prototype_like=left_zone_like,
            ),
            "pose_torso_frame": _pose_component_breakdown(
                sequence=sequence,
                pose_coord_start=pose_coord_start,
                pose_mask_start=pose_mask_start,
                pose_names=POSE_NAMES,
                selected_nodes=[0, 1, 2, 7, 8],
                prototype_reference=pose_torso_ref,
                prototype_like=pose_torso_like,
            ),
            "pose_arm_chain": _pose_component_breakdown(
                sequence=sequence,
                pose_coord_start=pose_coord_start,
                pose_mask_start=pose_mask_start,
                pose_names=POSE_NAMES,
                selected_nodes=[3, 4, 5, 6],
                prototype_reference=pose_arm_ref,
                prototype_like=pose_arm_like,
            ),
        }
        rows.append(token_row)

    ranked = sorted(
        (
            {
                "group_name": group_name,
                "mean_reference_minus_like_l2": _safe_float(np.mean(values)),
                "token_count": len(values),
            }
            for group_name, values in fine_group_deltas.items()
        ),
        key=lambda item: item["mean_reference_minus_like_l2"],
        reverse=True,
    )

    output_payload = {
        "spans_json": str(spans_path),
        "classification_json": str(classification_path),
        "cache_path": str(cache_path),
        "mirror_input": int(mirror_input),
        "prototype_groups_per_class": int(args.prototype_groups_per_class),
        "dominant_subgroup": ranked[0]["group_name"] if ranked else "",
        "subgroup_rankings": ranked,
        "prototype_paths": prototype_paths,
        "rows": rows,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
