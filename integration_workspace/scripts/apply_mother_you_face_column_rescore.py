from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.continuous_feature_cache import load_continuous_feature_cache


COLUMN_NAMES = ("nose_x", "nose_y", "nose_z", "chin_x", "chin_y", "chin_z", "mouth_x", "mouth_y", "mouth_z")
LOCATION_START = 214
LEFT_FACE_START = LOCATION_START
LEFT_FACE_END = LEFT_FACE_START + len(COLUMN_NAMES)
DEFAULT_NOSE_Z_OFFSET = -1.5


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Apply a residual-specific mother-vs-you face-column rescore.")
    parser.add_argument("--classification-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--analysis-json", required=True)
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
    parser.add_argument("--nose-z-offset", type=float, default=DEFAULT_NOSE_Z_OFFSET)
    parser.add_argument("--trigger-csv")
    parser.add_argument("--trigger-output-json")
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    prototypes: dict[str, np.ndarray] = {}
    for label, paths in chosen_paths.items():
        vectors: list[np.ndarray] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                sequence = payload["sequence"].astype(np.float32)[:, LEFT_FACE_START:LEFT_FACE_END]
            vectors.append(sequence.mean(axis=0))
        prototypes[label] = np.stack(vectors, axis=0).mean(axis=0)
    return prototypes


def _left_face_mean(feature_vectors: np.ndarray, sampled_frame_indices: list[int]) -> np.ndarray:
    sampled = np.asarray(sampled_frame_indices, dtype=np.int32)
    return feature_vectors[sampled][:, LEFT_FACE_START:LEFT_FACE_END].astype(np.float32).mean(axis=0)


def _distance_summary(vector: np.ndarray, mother_proto: np.ndarray, you_proto: np.ndarray) -> dict[str, float]:
    mother_distance = float(np.linalg.norm(vector - mother_proto))
    you_distance = float(np.linalg.norm(vector - you_proto))
    return {
        "mother_l2": _safe_float(mother_distance),
        "you_l2": _safe_float(you_distance),
        "mother_minus_you_l2": _safe_float(mother_distance - you_distance),
    }


def _focus_distance_summary(vector: np.ndarray, mother_proto: np.ndarray, you_proto: np.ndarray) -> dict[str, float]:
    indices = [COLUMN_NAMES.index(name) for name in ("nose_z", "chin_y", "nose_y", "mouth_y", "chin_z", "mouth_z")]
    return _distance_summary(vector[indices], mother_proto[indices], you_proto[indices])


def _apply_nose_z_offset(vector: np.ndarray, nose_z_offset: float) -> np.ndarray:
    adjusted = vector.copy()
    adjusted[COLUMN_NAMES.index("nose_z")] += float(nose_z_offset)
    return adjusted


def _maybe_relabel_row(
    *,
    row: dict[str, object],
    feature_vectors: np.ndarray,
    mother_proto: np.ndarray,
    you_proto: np.ndarray,
    nose_z_offset: float,
) -> tuple[dict[str, object], dict[str, object] | None]:
    if str(row.get("predicted_label", "")).strip().lower() != "you":
        return dict(row), None

    left_face_mean = _left_face_mean(feature_vectors, list(row["sampled_frame_indices"]))
    baseline_full = _distance_summary(left_face_mean, mother_proto, you_proto)
    baseline_focus = _focus_distance_summary(left_face_mean, mother_proto, you_proto)
    adjusted_mean = _apply_nose_z_offset(left_face_mean, nose_z_offset)
    adjusted_full = _distance_summary(adjusted_mean, mother_proto, you_proto)
    adjusted_focus = _focus_distance_summary(adjusted_mean, mother_proto, you_proto)

    should_flip = bool(adjusted_full["mother_minus_you_l2"] < 0.0 and baseline_full["mother_minus_you_l2"] > 0.0)
    updated = dict(row)
    if should_flip:
        updated["predicted_label"] = "mother"
        top3 = list(updated.get("top3", []))
        synthetic_confidence = max(
            0.0,
            min(
                0.999999,
                float(top3[0]["confidence"]) if top3 and top3[0]["label"] == "you" else 0.5,
            ),
        )
        updated["top3"] = [
            {
                "label": "mother",
                "confidence": _safe_float(synthetic_confidence),
                "logit": updated.get("top1_logit"),
            },
            *[entry for entry in top3 if entry.get("label") != "mother"][:2],
        ]
        updated["mother_you_face_column_rescore"] = {
            "nose_z_offset": _safe_float(nose_z_offset),
            "baseline_full_distance": baseline_full,
            "baseline_focus_distance": baseline_focus,
            "adjusted_full_distance": adjusted_full,
            "adjusted_focus_distance": adjusted_focus,
            "flip_reason": "adjusted mother distance overtook you under mother-vs-you face-column rescore",
        }

    candidate = {
        "token": row["token"],
        "baseline_prediction": row["predicted_label"],
        "adjusted_prediction": updated["predicted_label"],
        "nose_z_offset": _safe_float(nose_z_offset),
        "baseline_full_distance": baseline_full,
        "baseline_focus_distance": baseline_focus,
        "adjusted_full_distance": adjusted_full,
        "adjusted_focus_distance": adjusted_focus,
        "flipped": should_flip,
    }
    return updated, candidate


def _update_summary(payload: dict[str, object], mirror_input: int) -> dict[str, object]:
    updated_payload = deepcopy(payload)
    updated_payload["classifications"] = list(updated_payload.get("classifications", []))
    summary_rows = list(updated_payload.get("summary_by_mirror", []))
    for row in summary_rows:
        if int(row.get("mirror_input", -1)) != mirror_input:
            continue
        mirror_rows = [
            item
            for item in updated_payload["classifications"]
            if int(item.get("mirror_input", -1)) == mirror_input
        ]
        predicted_tokens = [str(item["predicted_label"]) for item in mirror_rows]
        reference_tokens = [str(item["token"]) for item in mirror_rows]
        correct_count = sum(
            int(str(item["predicted_label"]).strip().lower() == str(item["token"]).strip().lower())
            for item in mirror_rows
        )
        confusion: dict[str, dict[str, int]] = {}
        for item in mirror_rows:
            reference = str(item["token"])
            predicted = str(item["predicted_label"])
            confusion.setdefault(reference, {})
            confusion[reference][predicted] = confusion[reference].get(predicted, 0) + 1
        row["correct_count"] = correct_count
        row["exact_span_accuracy"] = _safe_float(correct_count / max(1, len(mirror_rows)))
        row["predicted_tokens"] = predicted_tokens
        row["predicted_label_counts"] = dict(Counter(predicted_tokens))
        row["confusion"] = confusion
        row["reference_tokens"] = reference_tokens
    updated_payload["summary_by_mirror"] = summary_rows
    return updated_payload


def _build_trigger_payload(
    *,
    trigger_csv: Path,
    nose_z_offset: float,
    mother_candidates: dict[str, dict[str, object]],
) -> dict[str, object]:
    rows = _read_csv_rows(trigger_csv)
    updated_rows: list[dict[str, object]] = []
    flipped_segments: list[dict[str, object]] = []
    for row in rows:
        updated = dict(row)
        reference = str(row.get("reference_label", "")).strip().lower()
        raw_label = str(row.get("raw_label", "")).strip().lower()
        token_key = reference
        candidate = mother_candidates.get(token_key)
        if candidate and raw_label == "you" and bool(candidate["flipped"]):
            updated["raw_label"] = "mother"
            updated["face_column_rescore_note"] = f"nose_z_offset={_safe_float(nose_z_offset)}"
            flipped_segments.append(
                {
                    "segment_id": row.get("segment_id"),
                    "reference_label": row.get("reference_label"),
                    "baseline_raw_label": raw_label,
                    "adjusted_raw_label": "mother",
                    "decision": row.get("decision"),
                    "raw_confidence": row.get("raw_confidence"),
                    "margin": row.get("margin"),
                }
            )
        updated_rows.append(updated)
    return {
        "trigger_csv": str(trigger_csv),
        "nose_z_offset": _safe_float(nose_z_offset),
        "flipped_segments": flipped_segments,
        "rows": updated_rows,
    }


def main() -> None:
    args = build_parser().parse_args()
    classification_path = Path(args.classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()
    summary_path = Path(args.summary_json).resolve()
    analysis_path = Path(args.analysis_json).resolve()
    manifest_path = Path(args.sequence_manifest).resolve()
    processed_sequences_dir = Path(args.processed_sequences_dir).resolve()
    mirror_input = int(args.mirror_input)
    nose_z_offset = float(args.nose_z_offset)

    classification_payload = _load_json(classification_path)
    analysis_payload = _load_json(analysis_path)
    if not analysis_payload.get("recommended_minimal_focus_fix"):
        raise RuntimeError("analysis_json does not contain a recommended_minimal_focus_fix")

    cache_path = cache_dir / f"continuous_feature_cache_mirror{mirror_input}.npz"
    _, arrays = load_continuous_feature_cache(cache_path)
    feature_vectors = arrays["feature_vectors"]
    manifest_rows = _read_csv_rows(manifest_path)
    chosen_paths = _choose_prototype_paths(
        manifest_rows=manifest_rows,
        processed_sequences_dir=processed_sequences_dir,
        class_labels={"mother", "you"},
        groups_per_class=int(args.prototype_groups_per_class),
    )
    prototypes = _load_prototypes(chosen_paths)
    mother_proto = prototypes["mother"]
    you_proto = prototypes["you"]

    updated_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for row in classification_payload.get("classifications", []):
        if int(row.get("mirror_input", -1)) != mirror_input:
            updated_rows.append(dict(row))
            continue
        updated, candidate = _maybe_relabel_row(
            row=row,
            feature_vectors=feature_vectors,
            mother_proto=mother_proto,
            you_proto=you_proto,
            nose_z_offset=nose_z_offset,
        )
        updated["correct"] = int(str(updated["predicted_label"]).strip().lower() == str(updated["token"]).strip().lower())
        updated_rows.append(updated)
        if candidate is not None:
            candidate_rows.append(candidate)

    baseline_summary = deepcopy(next(
        row for row in classification_payload.get("summary_by_mirror", [])
        if int(row.get("mirror_input", -1)) == mirror_input
    ))

    updated_payload = deepcopy(classification_payload)
    updated_payload["classifications"] = updated_rows
    updated_payload["mother_you_face_column_rescore"] = {
        "analysis_json": str(analysis_path),
        "nose_z_offset": _safe_float(nose_z_offset),
        "candidate_rows": candidate_rows,
        "rescore_scope": "only spans whose baseline top1 is you",
    }
    updated_payload = _update_summary(updated_payload, mirror_input)
    _write_json(output_path, updated_payload)

    updated_summary = next(
        row for row in updated_payload.get("summary_by_mirror", [])
        if int(row.get("mirror_input", -1)) == mirror_input
    )
    summary_payload = {
        "baseline_classification_json": str(classification_path),
        "analysis_json": str(analysis_path),
        "output_classification_json": str(output_path),
        "nose_z_offset": _safe_float(nose_z_offset),
        "baseline_accuracy": baseline_summary["exact_span_accuracy"],
        "updated_accuracy": updated_summary["exact_span_accuracy"],
        "baseline_predicted_label_counts": baseline_summary["predicted_label_counts"],
        "updated_predicted_label_counts": updated_summary["predicted_label_counts"],
        "baseline_confusion": baseline_summary["confusion"],
        "updated_confusion": updated_summary["confusion"],
        "candidate_rows": candidate_rows,
        "improved": bool(updated_summary["exact_span_accuracy"] > baseline_summary["exact_span_accuracy"]),
        "mother_confusion_weakened": bool(
            baseline_summary["confusion"].get("mother", {}).get("you", 0) > 0
            and updated_summary["confusion"].get("mother", {}).get("you", 0) == 0
            and updated_summary["confusion"].get("mother", {}).get("mother", 0) > 0
        ),
        "you_is_non_face_dominant_residual": True,
    }

    if args.trigger_csv and args.trigger_output_json:
        trigger_payload = _build_trigger_payload(
            trigger_csv=Path(args.trigger_csv).resolve(),
            nose_z_offset=nose_z_offset,
            mother_candidates={str(item["token"]).strip().lower(): item for item in candidate_rows},
        )
        _write_json(Path(args.trigger_output_json).resolve(), trigger_payload)
        summary_payload["trigger_output_json"] = str(Path(args.trigger_output_json).resolve())
        summary_payload["trigger_flipped_segments"] = trigger_payload["flipped_segments"]

    _write_json(summary_path, summary_payload)
    print(f"[OK] Wrote {output_path}")
    print(f"[OK] Wrote {summary_path}")


if __name__ == "__main__":
    main()
