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


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build per-span feature-group statistics with prototype distances.")
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
    motion_range = feature_spec["components"]["motion"]
    explicit_range = feature_spec["landmark_sections"]["explicit_finger_states"]

    location_start = int(location_range["start"])
    location_end = int(location_range["end"])
    location_validity_start = location_end - 4
    motion_start = int(motion_range["start"])
    motion_end = int(motion_range["end"])
    motion_validity_start = motion_end - 2
    explicit_start = int(explicit_range["start"])
    explicit_end = int(explicit_range["end"])

    return {
        "hand_local_graph": list(range(0, left_xyz + right_xyz)),
        "pose_context_graph": list(range(left_xyz + right_xyz, left_xyz + right_xyz + pose_xyz))
        + list(range(pose_mask_start, pose_mask_start + pose_nodes)),
        "location_vectors": list(range(location_start, location_validity_start)),
        "hand_mask_validity": list(range(left_mask_start, left_mask_start + int(CONFIG.left_hand_nodes)))
        + list(range(right_mask_start, right_mask_start + int(CONFIG.right_hand_nodes)))
        + list(range(location_validity_start, location_end))
        + list(range(motion_validity_start, motion_end)),
        "motion_core": list(range(motion_start, motion_validity_start)),
        "explicit_finger_state": list(range(explicit_start, explicit_end)),
        "pose_location_contextual": list(range(left_xyz + right_xyz, left_xyz + right_xyz + pose_xyz))
        + list(range(pose_mask_start, pose_mask_start + pose_nodes))
        + list(range(location_start, location_validity_start)),
    }


def _sequence_group_summary(sequence: np.ndarray, indices: list[int]) -> dict[str, object]:
    subset = sequence[:, indices].astype(np.float32)
    feature_mean_vector = subset.mean(axis=0)
    feature_std_vector = subset.std(axis=0)
    frame_energy = np.linalg.norm(subset, axis=1)
    peak_index = int(np.argmax(frame_energy)) if len(frame_energy) else 0
    peak_value = float(frame_energy[peak_index]) if len(frame_energy) else 0.0
    active_threshold = peak_value * 0.25 if peak_value > 0 else 0.0
    return {
        "feature_mean_vector": feature_mean_vector,
        "feature_std_vector": feature_std_vector,
        "feature_mean": _safe_float(np.mean(feature_mean_vector)),
        "feature_std": _safe_float(np.mean(feature_std_vector)),
        "feature_energy": _safe_float(np.mean(frame_energy ** 2)),
        "zero_ratio": _safe_float(np.mean(np.abs(subset) <= 1e-6)),
        "frame_activation_mean": _safe_float(np.mean(frame_energy)),
        "frame_activation_std": _safe_float(np.std(frame_energy)),
        "peak_frame_offset": peak_index,
        "peak_frame_energy": _safe_float(peak_value),
        "active_frame_fraction": _safe_float(np.mean(frame_energy >= active_threshold)) if peak_value > 0 else 0.0,
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
        group_scalar_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                sequence = payload["sequence"].astype(np.float32)
                for group_name, indices in group_indices.items():
                    summary = _sequence_group_summary(sequence, indices)
                    group_vectors[group_name].append(summary["feature_mean_vector"])
                    group_std_vectors[group_name].append(summary["feature_std_vector"])
                    for key in (
                        "feature_mean",
                        "feature_std",
                        "feature_energy",
                        "zero_ratio",
                        "frame_activation_mean",
                        "frame_activation_std",
                        "peak_frame_energy",
                        "active_frame_fraction",
                    ):
                        group_scalar_stats[group_name][key].append(float(summary[key]))
        prototypes[label] = {
            "sample_count": len(paths),
            "groups": {},
        }
        for group_name in group_indices:
            mean_vectors = group_vectors[group_name]
            std_vectors = group_std_vectors[group_name]
            if not mean_vectors:
                continue
            prototypes[label]["groups"][group_name] = {
                "prototype_mean_vector": np.stack(mean_vectors, axis=0).mean(axis=0),
                "prototype_std_vector": np.stack(std_vectors, axis=0).mean(axis=0),
                "feature_mean": _safe_float(np.mean(group_scalar_stats[group_name]["feature_mean"])),
                "feature_std": _safe_float(np.mean(group_scalar_stats[group_name]["feature_std"])),
                "feature_energy": _safe_float(np.mean(group_scalar_stats[group_name]["feature_energy"])),
                "zero_ratio": _safe_float(np.mean(group_scalar_stats[group_name]["zero_ratio"])),
                "frame_activation_mean": _safe_float(np.mean(group_scalar_stats[group_name]["frame_activation_mean"])),
                "frame_activation_std": _safe_float(np.mean(group_scalar_stats[group_name]["frame_activation_std"])),
                "peak_frame_energy": _safe_float(np.mean(group_scalar_stats[group_name]["peak_frame_energy"])),
                "active_frame_fraction": _safe_float(np.mean(group_scalar_stats[group_name]["active_frame_fraction"])),
            }
    return prototypes, chosen_serialized


def _distance_summary(
    span_summary: dict[str, object],
    prototype_summary: dict[str, object],
) -> dict[str, float]:
    span_mean = span_summary["feature_mean_vector"]
    prototype_mean = prototype_summary["prototype_mean_vector"]
    span_std = span_summary["feature_std_vector"]
    prototype_std = prototype_summary["prototype_std_vector"]
    return {
        "mean_vector_l2": _safe_float(np.linalg.norm(span_mean - prototype_mean)),
        "mean_vector_cosine": _safe_float(_cosine_distance(span_mean, prototype_mean)),
        "std_vector_l2": _safe_float(np.linalg.norm(span_std - prototype_std)),
        "energy_gap": _safe_float(abs(float(span_summary["feature_energy"]) - float(prototype_summary["feature_energy"]))),
        "zero_ratio_gap": _safe_float(abs(float(span_summary["zero_ratio"]) - float(prototype_summary["zero_ratio"]))),
    }


def _raw_validity_summary(arrays: dict[str, np.ndarray], start_frame: int, end_frame: int) -> dict[str, float]:
    span = slice(start_frame, end_frame + 1)
    left_valid = np.mean(np.sum(arrays["left_hand_mask"][span], axis=1) > 0)
    right_valid = np.mean(np.sum(arrays["right_hand_mask"][span], axis=1) > 0)
    pose_valid = np.mean(np.sum(arrays["pose_mask"][span], axis=1) > 0)
    mouth_valid = np.mean(arrays["mouth_mask"][span, 0] > 0)
    chin_valid = np.mean(arrays["chin_mask"][span, 0] > 0)
    return {
        "left_hand_valid_ratio": _safe_float(left_valid),
        "right_hand_valid_ratio": _safe_float(right_valid),
        "pose_valid_ratio": _safe_float(pose_valid),
        "mouth_valid_ratio": _safe_float(mouth_valid),
        "chin_valid_ratio": _safe_float(chin_valid),
    }


def _normalized_validity_summary(arrays: dict[str, np.ndarray], start_frame: int, end_frame: int) -> dict[str, float]:
    span = slice(start_frame, end_frame + 1)
    left_valid = np.mean(np.linalg.norm(arrays["normalized_left_hand"][span], axis=2).sum(axis=1) > 0)
    right_valid = np.mean(np.linalg.norm(arrays["normalized_right_hand"][span], axis=2).sum(axis=1) > 0)
    pose_valid = np.mean(np.linalg.norm(arrays["normalized_pose"][span], axis=2).sum(axis=1) > 0)
    mouth_valid = np.mean(np.linalg.norm(arrays["normalized_mouth_center"][span], axis=1) > 0)
    chin_valid = np.mean(np.linalg.norm(arrays["normalized_chin"][span], axis=1) > 0)
    return {
        "left_hand_valid_ratio": _safe_float(left_valid),
        "right_hand_valid_ratio": _safe_float(right_valid),
        "pose_valid_ratio": _safe_float(pose_valid),
        "mouth_valid_ratio": _safe_float(mouth_valid),
        "chin_valid_ratio": _safe_float(chin_valid),
    }


def _nearest_prototype(
    span_summary: dict[str, object],
    prototypes: dict[str, dict[str, object]],
    group_name: str,
) -> tuple[str, float]:
    best_label = ""
    best_distance = float("inf")
    for label, payload in prototypes.items():
        group_payload = payload["groups"].get(group_name)
        if group_payload is None:
            continue
        distance = np.linalg.norm(span_summary["feature_mean_vector"] - group_payload["prototype_mean_vector"])
        if float(distance) < best_distance:
            best_distance = float(distance)
            best_label = label
    return best_label, _safe_float(best_distance if best_label else 0.0)


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
    feature_spec = metadata["feature_spec"]
    group_indices = _group_slices(feature_spec)
    manifest_rows = _read_csv_rows(manifest_path)

    token_predictions = {
        str(row["token"]): row
        for row in classification_payload.get("classifications", [])
        if int(row.get("mirror_input", -1)) == int(mirror_input)
    }
    class_labels = {str(span["token"]).strip().lower() for span in spans_payload.get("token_span_hypothesis", [])}
    class_labels.update(str(row.get("predicted_label", "")).strip().lower() for row in token_predictions.values())
    class_labels.discard("")

    prototypes, prototype_paths = _build_class_prototypes(
        manifest_rows=manifest_rows,
        processed_sequences_dir=processed_sequences_dir,
        class_labels=class_labels,
        group_indices=group_indices,
        groups_per_class=int(args.prototype_groups_per_class),
    )

    feature_vectors = arrays["feature_vectors"].astype(np.float32)
    rows: list[dict[str, object]] = []
    families = {
        "like_family": {"i", "you", "mother"},
        "student_family": {"student", "want", "teacher", "father"},
    }
    family_group_deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for span in spans_payload.get("token_span_hypothesis", []):
        token = str(span["token"]).strip().lower()
        classification = token_predictions.get(token, {})
        predicted_label = str(classification.get("predicted_label", "")).strip().lower()
        start_frame = int(span["start_frame"])
        end_frame = int(span["end_frame"])
        sampled_indices = np.asarray(classification.get("sampled_frame_indices"), dtype=np.int32)
        if sampled_indices.size == 0:
            sampled_indices = (
                np.linspace(start_frame, end_frame, 30).round().astype(np.int32)
                if end_frame > start_frame
                else np.full((30,), start_frame, dtype=np.int32)
            )
        sampled_sequence = feature_vectors[sampled_indices]
        group_stats: dict[str, object] = {}
        for group_name, indices in group_indices.items():
            summary = _sequence_group_summary(sampled_sequence, indices)
            nearest_label, nearest_distance = _nearest_prototype(summary, prototypes, group_name)
            reference_payload = prototypes.get(token, {}).get("groups", {}).get(group_name)
            attractor_payload = prototypes.get(predicted_label, {}).get("groups", {}).get(group_name)
            reference_distance = _distance_summary(summary, reference_payload) if reference_payload else {}
            attractor_distance = _distance_summary(summary, attractor_payload) if attractor_payload else {}
            delta = 0.0
            if reference_distance and attractor_distance:
                delta = float(reference_distance["mean_vector_l2"]) - float(attractor_distance["mean_vector_l2"])
            for family_name, family_tokens in families.items():
                if token in family_tokens:
                    family_group_deltas[family_name][group_name].append(delta)
            group_stats[group_name] = {
                "feature_mean": summary["feature_mean"],
                "feature_std": summary["feature_std"],
                "feature_energy": summary["feature_energy"],
                "zero_ratio": summary["zero_ratio"],
                "frame_activation_mean": summary["frame_activation_mean"],
                "frame_activation_std": summary["frame_activation_std"],
                "peak_frame_offset": int(summary["peak_frame_offset"]),
                "peak_frame_energy": summary["peak_frame_energy"],
                "active_frame_fraction": summary["active_frame_fraction"],
                "nearest_prototype_label": nearest_label,
                "nearest_prototype_distance": nearest_distance,
                "reference_label": token,
                "predicted_label": predicted_label,
                "distance_to_reference": reference_distance,
                "distance_to_attractor": attractor_distance,
                "reference_minus_attractor_l2": _safe_float(delta),
            }
        family_name = next((name for name, tokens in families.items() if token in tokens), "other")
        rows.append(
            {
                "token": token,
                "family": family_name,
                "predicted_label": predicted_label,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "sampled_frame_indices": sampled_indices.astype(int).tolist(),
                "raw_validity": _raw_validity_summary(arrays, start_frame, end_frame),
                "normalized_validity": _normalized_validity_summary(arrays, start_frame, end_frame),
                "group_stats": group_stats,
            }
        )

    family_summary: dict[str, object] = {}
    for family_name, group_payload in family_group_deltas.items():
        ranked = sorted(
            (
                {
                    "group_name": group_name,
                    "mean_reference_minus_attractor_l2": _safe_float(np.mean(values)),
                    "token_count": len(values),
                }
                for group_name, values in group_payload.items()
            ),
            key=lambda item: item["mean_reference_minus_attractor_l2"],
            reverse=True,
        )
        family_summary[family_name] = {
            "group_rankings": ranked,
            "dominant_group": ranked[0]["group_name"] if ranked else "",
        }

    serialized_prototypes = {}
    for label, payload in prototypes.items():
        serialized_prototypes[label] = {
            "sample_count": payload["sample_count"],
            "sample_paths": prototype_paths.get(label, []),
            "groups": {
                group_name: {
                    key: value
                    for key, value in group_payload.items()
                    if key not in {"prototype_mean_vector", "prototype_std_vector"}
                }
                for group_name, group_payload in payload["groups"].items()
            },
        }

    output_payload = {
        "spans_json": str(spans_path),
        "classification_json": str(classification_path),
        "cache_path": str(cache_path),
        "sequence_manifest": str(manifest_path),
        "processed_sequences_dir": str(processed_sequences_dir),
        "mirror_input": int(mirror_input),
        "group_names": list(group_indices.keys()),
        "family_summary": family_summary,
        "prototype_summary": serialized_prototypes,
        "rows": rows,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
