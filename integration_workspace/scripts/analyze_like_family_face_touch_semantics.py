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

from src.dataset.continuous_feature_cache import load_continuous_feature_cache
from src.landmarks.location_features import build_location_features


EPS = 1e-8
FACE_REFERENCE_NAMES = ("nose", "chin", "mouth_center")
FACE_ANCHOR_MODES = ("center", "thumb_tip", "index_tip", "middle_tip", "closest_valid_fingertip")
PAIR_CONFIG = (
    ("you", "you", "i"),
    ("mother", "mother", "you"),
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Analyze torso-center residual confusion inside left_face_vectors.")
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


def _raw_validity_summary(arrays: dict[str, np.ndarray], start_frame: int, end_frame: int) -> dict[str, float]:
    span = slice(start_frame, end_frame + 1)
    left_valid = np.mean(np.sum(arrays["left_hand_mask"][span], axis=1) > 0)
    pose_valid = np.mean(np.sum(arrays["pose_mask"][span], axis=1) > 0)
    mouth_valid = np.mean(arrays["mouth_mask"][span, 0] > 0)
    chin_valid = np.mean(arrays["chin_mask"][span, 0] > 0)
    return {
        "left_hand_valid_ratio": _safe_float(left_valid),
        "pose_valid_ratio": _safe_float(pose_valid),
        "mouth_valid_ratio": _safe_float(mouth_valid),
        "chin_valid_ratio": _safe_float(chin_valid),
    }


def _normalized_validity_summary(arrays: dict[str, np.ndarray], start_frame: int, end_frame: int) -> dict[str, float]:
    span = slice(start_frame, end_frame + 1)
    left_valid = np.mean(np.linalg.norm(arrays["normalized_left_hand"][span], axis=2).sum(axis=1) > 0)
    pose_valid = np.mean(np.linalg.norm(arrays["normalized_pose"][span], axis=2).sum(axis=1) > 0)
    mouth_valid = np.mean(np.linalg.norm(arrays["normalized_mouth_center"][span], axis=1) > 0)
    chin_valid = np.mean(np.linalg.norm(arrays["normalized_chin"][span], axis=1) > 0)
    return {
        "left_hand_valid_ratio": _safe_float(left_valid),
        "pose_valid_ratio": _safe_float(pose_valid),
        "mouth_valid_ratio": _safe_float(mouth_valid),
        "chin_valid_ratio": _safe_float(chin_valid),
    }


def _choose_prototype_paths(
    *,
    manifest_rows: list[dict[str, str]],
    processed_sequences_dir: Path,
    class_labels: set[str],
    groups_per_class: int,
) -> dict[str, list[Path]]:
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
        sample_path = processed_sequences_dir / f"{row['sample_id'].strip()}.npz"
        if not sample_path.exists():
            continue
        chosen_paths[label].append(sample_path)
        seen_groups[label].add(group_key)
    return chosen_paths


def _load_face_prototypes(
    chosen_paths: dict[str, list[Path]],
) -> dict[str, dict[str, object]]:
    prototypes: dict[str, dict[str, object]] = {}
    location_start = 214
    for label, paths in chosen_paths.items():
        if not paths:
            continue
        mean_vectors: list[np.ndarray] = []
        std_vectors: list[np.ndarray] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                sequence = payload["sequence"].astype(np.float32)[:, location_start : location_start + 9]
            mean_vectors.append(sequence.mean(axis=0))
            std_vectors.append(sequence.std(axis=0))
        prototypes[label] = {
            "sample_count": len(paths),
            "sample_paths": [str(path) for path in paths],
            "mean_vector": np.stack(mean_vectors, axis=0).mean(axis=0),
            "std_vector": np.stack(std_vectors, axis=0).mean(axis=0),
        }
    return prototypes


def _frame_feature(arrays: dict[str, np.ndarray], frame_index: int) -> dict[str, np.ndarray]:
    left_mask = arrays["left_hand_mask"][frame_index].astype(np.float32)
    right_mask = arrays["right_hand_mask"][frame_index].astype(np.float32)
    return {
        "left_hand": arrays["normalized_left_hand"][frame_index].astype(np.float32),
        "right_hand": arrays["normalized_right_hand"][frame_index].astype(np.float32),
        "left_hand_mask": left_mask,
        "right_hand_mask": right_mask,
        "pose_mask": arrays["pose_mask"][frame_index].astype(np.float32),
        "mouth_mask": arrays["mouth_mask"][frame_index].astype(np.float32),
        "chin_mask": arrays["chin_mask"][frame_index].astype(np.float32),
        "left_hand_center": arrays["normalized_left_hand_center"][frame_index].astype(np.float32),
        "right_hand_center": arrays["normalized_right_hand_center"][frame_index].astype(np.float32),
        "left_hand_valid": np.array([float(np.any(left_mask > 0))], dtype=np.float32),
        "right_hand_valid": np.array([float(np.any(right_mask > 0))], dtype=np.float32),
        "reference_points": {
            "nose": arrays["normalized_nose"][frame_index].astype(np.float32),
            "chin": arrays["normalized_chin"][frame_index].astype(np.float32),
            "mouth_center": arrays["normalized_mouth_center"][frame_index].astype(np.float32),
            "shoulder_center": arrays["normalized_shoulder_center"][frame_index].astype(np.float32),
            "chest_center": arrays["normalized_chest_center"][frame_index].astype(np.float32),
            "torso_center": arrays["normalized_torso_center"][frame_index].astype(np.float32),
        },
    }


def _build_left_face_sequence(
    *,
    arrays: dict[str, np.ndarray],
    sampled_indices: np.ndarray,
    reference_modes: tuple[str, str, str],
    anchor_modes: tuple[str, str, str],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for frame_index in sampled_indices.astype(int).tolist():
        location = build_location_features(
            _frame_feature(arrays, frame_index),
            left_anchor_mode="center",
            right_anchor_mode="center",
            left_face_reference_modes=reference_modes,
            left_face_anchor_modes=anchor_modes,
        )["feature_vector"].astype(np.float32)
        rows.append(location[:9])
    return np.stack(rows, axis=0)


def _distance_summary(mean_vector: np.ndarray, prototype_vector: np.ndarray) -> dict[str, float]:
    return {
        "mean_vector_l2": _safe_float(np.linalg.norm(mean_vector - prototype_vector)),
        "mean_vector_cosine": _safe_float(_cosine_distance(mean_vector, prototype_vector)),
    }


def _prefix_takeover(sequence: np.ndarray, reference_proto: np.ndarray, confused_proto: np.ndarray) -> dict[str, object]:
    prefix_means = np.cumsum(sequence, axis=0) / np.arange(1, sequence.shape[0] + 1, dtype=np.float32)[:, None]
    deltas: list[float] = []
    first_positive = -1
    for offset, prefix_mean in enumerate(prefix_means):
        delta = float(np.linalg.norm(prefix_mean - reference_proto) - np.linalg.norm(prefix_mean - confused_proto))
        deltas.append(_safe_float(delta))
        if delta > 0 and first_positive < 0:
            first_positive = offset
    return {
        "first_true_to_confused_takeover_offset": first_positive,
        "prefix_true_minus_confused_l2": deltas,
    }


def _per_reference_breakdown(
    sequence: np.ndarray,
    reference_proto: np.ndarray,
    confused_proto: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(("nose", "chin", "mouth")):
        start = index * 3
        end = start + 3
        mean_vector = sequence[:, start:end].mean(axis=0)
        true_l2 = float(np.linalg.norm(mean_vector - reference_proto[start:end]))
        confused_l2 = float(np.linalg.norm(mean_vector - confused_proto[start:end]))
        takeover = _prefix_takeover(sequence[:, start:end], reference_proto[start:end], confused_proto[start:end])
        rows.append(
            {
                "name": name,
                "mean_vector": [_safe_float(value) for value in mean_vector.tolist()],
                "feature_mean": _safe_float(np.mean(sequence[:, start:end])),
                "feature_std": _safe_float(np.std(sequence[:, start:end])),
                "reference_l2": _safe_float(true_l2),
                "confused_l2": _safe_float(confused_l2),
                "true_minus_confused_l2": _safe_float(true_l2 - confused_l2),
                "takeover": takeover,
            }
        )
    rows.sort(key=lambda item: item["true_minus_confused_l2"], reverse=True)
    return rows


def _per_dimension_stats(
    sequence: np.ndarray,
    reference_proto: np.ndarray,
    confused_proto: np.ndarray,
) -> list[dict[str, object]]:
    labels = (
        "nose_x",
        "nose_y",
        "nose_z",
        "chin_x",
        "chin_y",
        "chin_z",
        "mouth_x",
        "mouth_y",
        "mouth_z",
    )
    rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        values = sequence[:, index].astype(np.float32)
        ref_gap = abs(float(values.mean()) - float(reference_proto[index]))
        confused_gap = abs(float(values.mean()) - float(confused_proto[index]))
        rows.append(
            {
                "name": label,
                "mean": _safe_float(values.mean()),
                "std": _safe_float(values.std()),
                "reference_value": _safe_float(reference_proto[index]),
                "confused_value": _safe_float(confused_proto[index]),
                "reference_gap": _safe_float(ref_gap),
                "confused_gap": _safe_float(confused_gap),
                "true_minus_confused_gap": _safe_float(ref_gap - confused_gap),
            }
        )
    rows.sort(key=lambda item: item["true_minus_confused_gap"], reverse=True)
    return rows


def _strategy_rows(
    *,
    arrays: dict[str, np.ndarray],
    sampled_indices: np.ndarray,
    reference_proto: np.ndarray,
    confused_proto: np.ndarray,
    true_label: str,
    confused_label: str,
) -> dict[str, object]:
    best_true_margin: dict[str, object] | None = None
    best_confused_pull: dict[str, object] | None = None
    true_favoring_strategies: list[dict[str, object]] = []
    searched = 0
    true_favoring_count = 0
    for nose_ref in FACE_REFERENCE_NAMES:
        for chin_ref in FACE_REFERENCE_NAMES:
            for mouth_ref in FACE_REFERENCE_NAMES:
                reference_modes = (nose_ref, chin_ref, mouth_ref)
                for nose_anchor in FACE_ANCHOR_MODES:
                    for chin_anchor in FACE_ANCHOR_MODES:
                        for mouth_anchor in FACE_ANCHOR_MODES:
                            anchor_modes = (nose_anchor, chin_anchor, mouth_anchor)
                            searched += 1
                            sequence = _build_left_face_sequence(
                                arrays=arrays,
                                sampled_indices=sampled_indices,
                                reference_modes=reference_modes,
                                anchor_modes=anchor_modes,
                            )
                            mean_vector = sequence.mean(axis=0)
                            true_l2 = float(np.linalg.norm(mean_vector - reference_proto))
                            confused_l2 = float(np.linalg.norm(mean_vector - confused_proto))
                            margin = float(true_l2 - confused_l2)
                            row = {
                                "reference_modes": list(reference_modes),
                                "anchor_modes": list(anchor_modes),
                                "true_l2": _safe_float(true_l2),
                                "confused_l2": _safe_float(confused_l2),
                                "true_minus_confused_l2": _safe_float(margin),
                            }
                            if margin <= 0:
                                true_favoring_count += 1
                            if best_true_margin is None or margin < float(best_true_margin["true_minus_confused_l2"]):
                                best_true_margin = row
                            if best_confused_pull is None or margin > float(best_confused_pull["true_minus_confused_l2"]):
                                best_confused_pull = row
                            if margin <= 0:
                                true_favoring_strategies.append(row)
    true_favoring_strategies.sort(key=lambda item: item["true_minus_confused_l2"])
    return {
        "searched_strategy_count": searched,
        "true_favoring_strategy_count": true_favoring_count,
        "best_true_margin_strategy": best_true_margin,
        "best_confused_pull_strategy": best_confused_pull,
        "true_favoring_strategies": true_favoring_strategies[:10],
        "conclusion": (
            f"No face-only strategy makes {true_label} closer to its true prototype than {confused_label}."
            if not true_favoring_strategies
            else (
                f"{len(true_favoring_strategies)} face-only strategies favor {true_label} over {confused_label}, "
                "but they still need exact-span validation."
            )
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    spans_path = Path(args.spans_json).resolve()
    classification_path = Path(args.classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()
    manifest_path = Path(args.sequence_manifest).resolve()
    processed_sequences_dir = Path(args.processed_sequences_dir).resolve()

    spans_payload = _load_json(spans_path)
    classification_payload = _load_json(classification_path)
    cache_path = cache_dir / f"continuous_feature_cache_mirror{int(bool(args.mirror_input))}.npz"
    _, arrays = load_continuous_feature_cache(cache_path)
    manifest_rows = _read_csv_rows(manifest_path)
    chosen_paths = _choose_prototype_paths(
        manifest_rows=manifest_rows,
        processed_sequences_dir=processed_sequences_dir,
        class_labels={"i", "you", "mother"},
        groups_per_class=int(args.prototype_groups_per_class),
    )
    prototypes = _load_face_prototypes(chosen_paths)
    classification_rows = {
        str(row["token"]).strip().lower(): row
        for row in classification_payload.get("classifications", [])
        if int(row.get("mirror_input", -1)) == int(bool(args.mirror_input))
    }
    spans = {str(row["token"]).strip().lower(): row for row in spans_payload.get("token_span_hypothesis", [])}

    pair_rows: list[dict[str, object]] = []
    pair_summary: dict[str, object] = {}
    for token, true_label, confused_label in PAIR_CONFIG:
        classification = classification_rows[token]
        sampled_indices = np.asarray(classification["sampled_frame_indices"], dtype=np.int32)
        sequence = _build_left_face_sequence(
            arrays=arrays,
            sampled_indices=sampled_indices,
            reference_modes=("nose", "chin", "mouth_center"),
            anchor_modes=("center", "center", "center"),
        )
        mean_vector = sequence.mean(axis=0)
        true_proto = prototypes[true_label]["mean_vector"]
        confused_proto = prototypes[confused_label]["mean_vector"]
        reference_breakdown = _per_reference_breakdown(sequence, true_proto, confused_proto)
        dimension_breakdown = _per_dimension_stats(sequence, true_proto, confused_proto)
        search_summary = _strategy_rows(
            arrays=arrays,
            sampled_indices=sampled_indices,
            reference_proto=true_proto,
            confused_proto=confused_proto,
            true_label=true_label,
            confused_label=confused_label,
        )
        dominant_reference = reference_breakdown[0]["name"] if reference_breakdown else ""
        pair_rows.append(
            {
                "token": token,
                "predicted_label": classification["predicted_label"],
                "true_label": true_label,
                "confused_label": confused_label,
                "start_frame": int(spans[token]["start_frame"]),
                "end_frame": int(spans[token]["end_frame"]),
                "sampled_frame_indices": sampled_indices.astype(int).tolist(),
                "raw_validity": _raw_validity_summary(arrays, int(spans[token]["start_frame"]), int(spans[token]["end_frame"])),
                "normalized_validity": _normalized_validity_summary(
                    arrays,
                    int(spans[token]["start_frame"]),
                    int(spans[token]["end_frame"]),
                ),
                "left_face_vectors": {
                    "current_reference_modes": ["nose", "chin", "mouth_center"],
                    "current_anchor_modes": ["center", "center", "center"],
                    "feature_mean": _safe_float(np.mean(mean_vector)),
                    "feature_std": _safe_float(np.std(sequence)),
                    "distance_to_true": _distance_summary(mean_vector, true_proto),
                    "distance_to_confused": _distance_summary(mean_vector, confused_proto),
                    "true_minus_confused_l2": _safe_float(
                        np.linalg.norm(mean_vector - true_proto) - np.linalg.norm(mean_vector - confused_proto)
                    ),
                    "per_reference": reference_breakdown,
                    "per_dimension": dimension_breakdown,
                },
                "dominant_reference": dominant_reference,
                "dominant_dimension": dimension_breakdown[0]["name"] if dimension_breakdown else "",
                "strategy_search": search_summary,
            }
        )
        pair_summary[f"{token}_vs_{confused_label}"] = {
            "dominant_reference": dominant_reference,
            "dominant_dimension": dimension_breakdown[0]["name"] if dimension_breakdown else "",
            "true_minus_confused_l2": _safe_float(
                np.linalg.norm(mean_vector - true_proto) - np.linalg.norm(mean_vector - confused_proto)
            ),
            "search_conclusion": search_summary["conclusion"],
        }

    output_payload = {
        "spans_json": str(spans_path),
        "classification_json": str(classification_path),
        "cache_path": str(cache_path),
        "prototype_source_paths": {label: [str(path) for path in paths] for label, paths in chosen_paths.items()},
        "pair_summary": pair_summary,
        "rows": pair_rows,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
