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


COLUMN_NAMES = ("nose_x", "nose_y", "nose_z", "chin_x", "chin_y", "chin_z", "mouth_x", "mouth_y", "mouth_z")
FOCUS_COLUMNS = ("nose_z", "chin_y", "nose_y", "mouth_y", "chin_z", "mouth_z")
FOCUS_COLUMN_INDICES = [COLUMN_NAMES.index(name) for name in FOCUS_COLUMNS]
DEFAULT_NOSE_Z_OFFSETS = (-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Analyze mother->you residual confusion at left_face_vectors column level.")
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
        seen_groups[label].add(group_key)
        chosen_paths[label].append(sample_path)
    return chosen_paths


def _load_prototypes(chosen_paths: dict[str, list[Path]]) -> dict[str, np.ndarray]:
    location_start = 214
    prototypes: dict[str, np.ndarray] = {}
    for label, paths in chosen_paths.items():
        vectors: list[np.ndarray] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                sequence = payload["sequence"].astype(np.float32)[:, location_start : location_start + 9]
            vectors.append(sequence.mean(axis=0))
        prototypes[label] = np.stack(vectors, axis=0).mean(axis=0)
    return prototypes


def _column_rows(
    *,
    mean_vector: np.ndarray,
    std_vector: np.ndarray,
    true_proto: np.ndarray,
    confused_proto: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(COLUMN_NAMES):
        true_gap = abs(float(mean_vector[index]) - float(true_proto[index]))
        confused_gap = abs(float(mean_vector[index]) - float(confused_proto[index]))
        rows.append(
            {
                "name": name,
                "mean": _safe_float(mean_vector[index]),
                "std": _safe_float(std_vector[index]),
                "energy": _safe_float(np.mean(mean_vector[index] ** 2)),
                "signed_delta_to_true": _safe_float(float(mean_vector[index]) - float(true_proto[index])),
                "signed_delta_to_confused": _safe_float(float(mean_vector[index]) - float(confused_proto[index])),
                "distance_to_true": _safe_float(true_gap),
                "distance_to_confused": _safe_float(confused_gap),
                "true_minus_confused_gap": _safe_float(true_gap - confused_gap),
            }
        )
    rows.sort(key=lambda item: item["true_minus_confused_gap"], reverse=True)
    return rows


def _subset_distance(mean_vector: np.ndarray, true_proto: np.ndarray, confused_proto: np.ndarray, indices: list[int]) -> dict[str, float]:
    mean_subset = mean_vector[indices]
    true_subset = true_proto[indices]
    confused_subset = confused_proto[indices]
    true_l2 = float(np.linalg.norm(mean_subset - true_subset))
    confused_l2 = float(np.linalg.norm(mean_subset - confused_subset))
    return {
        "true_l2": _safe_float(true_l2),
        "confused_l2": _safe_float(confused_l2),
        "true_minus_confused_l2": _safe_float(true_l2 - confused_l2),
    }


def _takeover_order(column_rows: list[dict[str, object]], mean_vector: np.ndarray, true_proto: np.ndarray, confused_proto: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected: list[int] = []
    for item in column_rows:
        selected.append(COLUMN_NAMES.index(str(item["name"])))
        distance = _subset_distance(mean_vector, true_proto, confused_proto, selected)
        rows.append(
            {
                "column_group": [COLUMN_NAMES[index] for index in selected],
                **distance,
            }
        )
    return rows


def _nose_z_offset_scan(
    *,
    mean_vector: np.ndarray,
    true_proto: np.ndarray,
    confused_proto: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    nose_z_index = COLUMN_NAMES.index("nose_z")
    for offset in DEFAULT_NOSE_Z_OFFSETS:
        adjusted = mean_vector.copy()
        adjusted[nose_z_index] += float(offset)
        full_distance = _subset_distance(adjusted, true_proto, confused_proto, list(range(len(COLUMN_NAMES))))
        focus_distance = _subset_distance(adjusted, true_proto, confused_proto, FOCUS_COLUMN_INDICES)
        rows.append(
            {
                "nose_z_offset": _safe_float(offset),
                "full_vector": full_distance,
                "focus_columns": focus_distance,
                "focus_favors_true": bool(float(focus_distance["true_minus_confused_l2"]) < 0.0),
            }
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    spans_path = Path(args.spans_json).resolve()
    classification_path = Path(args.classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()
    manifest_path = Path(args.sequence_manifest).resolve()
    processed_sequences_dir = Path(args.processed_sequences_dir).resolve()
    mirror_input = int(bool(args.mirror_input))

    spans_payload = _load_json(spans_path)
    classification_payload = _load_json(classification_path)
    cache_path = cache_dir / f"continuous_feature_cache_mirror{mirror_input}.npz"
    _, arrays = load_continuous_feature_cache(cache_path)
    manifest_rows = _read_csv_rows(manifest_path)
    chosen_paths = _choose_prototype_paths(
        manifest_rows=manifest_rows,
        processed_sequences_dir=processed_sequences_dir,
        class_labels={"mother", "you"},
        groups_per_class=int(args.prototype_groups_per_class),
    )
    prototypes = _load_prototypes(chosen_paths)

    span_rows = {str(row["token"]).strip().lower(): row for row in spans_payload.get("token_span_hypothesis", [])}
    classification_rows = {
        str(row["token"]).strip().lower(): row
        for row in classification_payload.get("classifications", [])
        if int(row.get("mirror_input", -1)) == mirror_input
    }
    mother_row = classification_rows["mother"]
    mother_indices = np.asarray(mother_row["sampled_frame_indices"], dtype=np.int32)
    mother_sequence = arrays["feature_vectors"][mother_indices][:, 214:223].astype(np.float32)
    mother_mean = mother_sequence.mean(axis=0)
    mother_std = mother_sequence.std(axis=0)
    true_proto = prototypes["mother"]
    confused_proto = prototypes["you"]

    column_rows = _column_rows(
        mean_vector=mother_mean,
        std_vector=mother_std,
        true_proto=true_proto,
        confused_proto=confused_proto,
    )
    takeover_rows = _takeover_order(column_rows, mother_mean, true_proto, confused_proto)
    nose_z_scan = _nose_z_offset_scan(mean_vector=mother_mean, true_proto=true_proto, confused_proto=confused_proto)
    successful_offsets = [row for row in nose_z_scan if row["focus_favors_true"]]
    minimal_focus_fix = min(
        successful_offsets,
        key=lambda row: (abs(float(row["nose_z_offset"])), float(row["focus_columns"]["true_minus_confused_l2"])),
    ) if successful_offsets else None

    output_payload = {
        "spans_json": str(spans_path),
        "classification_json": str(classification_path),
        "cache_path": str(cache_path),
        "prototype_source_paths": {label: [str(path) for path in paths] for label, paths in chosen_paths.items()},
        "token": "mother",
        "baseline_prediction": mother_row["predicted_label"],
        "sampled_frame_indices": mother_indices.astype(int).tolist(),
        "focus_columns": list(FOCUS_COLUMNS),
        "full_vector_distance": _subset_distance(mother_mean, true_proto, confused_proto, list(range(len(COLUMN_NAMES)))),
        "focus_column_distance": _subset_distance(mother_mean, true_proto, confused_proto, FOCUS_COLUMN_INDICES),
        "column_rows": column_rows,
        "takeover_order": takeover_rows,
        "nose_z_offset_scan": nose_z_scan,
        "recommended_minimal_focus_fix": minimal_focus_fix,
        "conclusion": {
            "dominant_column": column_rows[0]["name"] if column_rows else "",
            "dominant_small_group": list(FOCUS_COLUMNS[:3]),
            "root_cause_hypothesis": (
                "mother->you is dominated by left_face_vectors nose_z overshoot under torso_center-normalized face-touch semantics, "
                "with secondary y/z co-drift; "
                "a pure global column transform is insufficient, so any positive patch must stay mother-like residual specific."
            ),
        },
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
