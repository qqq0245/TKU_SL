from __future__ import annotations

import json
import random
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CONFIG
from src.landmarks.feature_builder import get_feature_spec, resolve_feature_spec
from src.models.feature_slices import get_feature_slices


def load_index(index_path: Path) -> list[dict]:
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")
    records = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def create_label_mapping(records: list[dict]) -> dict[str, int]:
    class_ids = [record.get("class_id") for record in records if record.get("class_id") is not None]
    if class_ids:
        mapping = {}
        for record in records:
            label = record["class_label"]
            class_id = record.get("class_id")
            if class_id is not None:
                mapping[label] = int(class_id)
        return dict(sorted(mapping.items(), key=lambda item: item[1]))
    labels = sorted({record["class_label"] for record in records})
    return {label: idx for idx, label in enumerate(labels)}


def split_records(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["class_label"], []).append(record)

    train_records = []
    val_records = []
    for label_records in grouped.values():
        rng.shuffle(label_records)
        if len(label_records) == 1:
            train_records.extend(label_records)
            continue
        val_count = max(1, int(round(len(label_records) * val_ratio)))
        if val_count >= len(label_records):
            val_count = 1
        val_records.extend(label_records[:val_count])
        train_records.extend(label_records[val_count:])
    return train_records, val_records


def _sample_translation_offset(rng: random.Random) -> tuple[float, float]:
    magnitude_x = rng.uniform(CONFIG.spatial_translation_min_offset, CONFIG.spatial_translation_max_offset)
    magnitude_y = rng.uniform(CONFIG.spatial_translation_min_offset, CONFIG.spatial_translation_max_offset)
    offset_x = magnitude_x * (1.0 if rng.random() >= 0.5 else -1.0)
    offset_y = magnitude_y * (1.0 if rng.random() >= 0.5 else -1.0)
    return float(offset_x), float(offset_y)


def _apply_spatial_translation_to_sequence(
    sequence: np.ndarray,
    *,
    feature_mode: str,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    augmented = sequence.copy()
    slices = get_feature_slices(feature_mode, total_dim=int(sequence.shape[-1]))
    location_slice = slices.location_stream
    if location_slice is None:
        return augmented

    location_stream = augmented[:, location_slice]
    # The first 12 vectors encode per-hand position against body/face anchors.
    # Apply the same sequence-level XY offset to keep temporal continuity while
    # making the spatial branch less brittle to position changes.
    vector_dim = 12 * 3
    if location_stream.shape[-1] < vector_dim:
        return augmented
    vectors = location_stream[:, :vector_dim].reshape(location_stream.shape[0], 12, 3)
    vectors[:, :, 0] += offset_x
    vectors[:, :, 1] += offset_y
    location_stream[:, :vector_dim] = vectors.reshape(location_stream.shape[0], vector_dim)
    augmented[:, location_slice] = location_stream
    return augmented.astype(np.float32)


class SignSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        records: list[dict],
        label_to_index: dict[str, int],
        *,
        training: bool = False,
        use_spatial_translation_aug: bool | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.records = records
        self.label_to_index = label_to_index
        self.training = training
        self.use_spatial_translation_aug = (
            CONFIG.use_spatial_translation_aug
            if use_spatial_translation_aug is None
            else bool(use_spatial_translation_aug)
        )
        self.rng = random.Random(CONFIG.random_seed if random_seed is None else random_seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        sample_path = Path(record.get("sample_path", record.get("path")))
        if not sample_path.is_absolute():
            sample_path = self.data_dir / sample_path
        data = np.load(sample_path, allow_pickle=True)
        sequence = data["sequence"].astype(np.float32)
        feature_mode = infer_record_feature_mode(record)
        if self.training and self.use_spatial_translation_aug:
            offset_x, offset_y = _sample_translation_offset(self.rng)
            sequence = _apply_spatial_translation_to_sequence(
                sequence,
                feature_mode=feature_mode,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        label = self.label_to_index[record["class_label"]]
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)


def load_split_records(split_csv_path: Path) -> list[dict]:
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")
    with split_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            record = {
                "sample_id": row.get("sample_id"),
                "class_label": row.get("english_label") or row.get("class_label"),
                "class_id": int(row["class_id"]) if row.get("class_id") not in {None, ""} else None,
                "sample_path": row.get("sample_path") or row.get("path"),
                "path": row.get("path") or Path(row.get("sample_path", "")).name,
                "feature_mode": row.get("feature_mode"),
                "sequence_length": int(row["sequence_length"]) if row.get("sequence_length") else None,
                "feature_dim": int(row["feature_dim"]) if row.get("feature_dim") else None,
                "metadata": {
                    "zh_tw_translation": row.get("zh_tw_translation"),
                    "source_video_path": row.get("source_video_path"),
                    "feature_mode": row.get("feature_mode"),
                },
            }
            records.append(record)
        return records


def validate_feature_mode(records: list[dict], expected_mode: str) -> None:
    inferred_modes = set()
    inferred_total_dims = set()
    expected_spec = resolve_feature_spec(feature_mode=expected_mode)
    for record in records:
        mode = record.get("feature_mode")
        if mode is None:
            metadata_mode = record.get("metadata", {}).get("feature_mode")
            if metadata_mode is not None:
                mode = metadata_mode
            else:
                feature_dim = record.get("feature_dim")
                if feature_dim == get_feature_spec("landmarks_only")["total_dim"]:
                    mode = "landmarks_only"
                elif feature_dim == get_feature_spec("landmarks_plus_location")["total_dim"]:
                    mode = "landmarks_plus_location"
                elif feature_dim == get_feature_spec("landmarks_plus_location_motion")["total_dim"]:
                    mode = "landmarks_plus_location_motion"
                else:
                    mode = "unknown"
        inferred_modes.add(mode)
        if mode == expected_mode:
            feature_dim = record.get("feature_dim")
            if feature_dim is not None:
                inferred_total_dims.add(int(feature_dim))

    if len(inferred_modes) > 1:
        raise RuntimeError(f"Mixed feature modes found in dataset: {sorted(str(mode) for mode in inferred_modes)}")

    dataset_mode = next(iter(inferred_modes))
    if dataset_mode != expected_mode:
        raise RuntimeError(
            "Dataset feature mode mismatch: "
            f"dataset_mode={dataset_mode}, expected_mode={expected_mode}. "
            "If this is old stage-1 data, either set config.feature_mode to 'landmarks_only' "
            "or recollect data with the new feature pipeline."
        )
    if inferred_total_dims and inferred_total_dims != {expected_spec["total_dim"]}:
        raise RuntimeError(
            "Dataset feature spec mismatch: "
            f"expected_total_dim={expected_spec['total_dim']} but found {sorted(inferred_total_dims)}. "
            "Re-export sequences with the current feature pipeline or filter to a compatible dataset version."
        )


def infer_record_feature_mode(record: dict) -> str:
    mode = record.get("feature_mode") or record.get("metadata", {}).get("feature_mode")
    if mode is not None:
        return mode
    feature_dim = record.get("feature_dim")
    if feature_dim == get_feature_spec("landmarks_only")["total_dim"]:
        return "landmarks_only"
    if feature_dim == get_feature_spec("landmarks_plus_location")["total_dim"]:
        return "landmarks_plus_location"
    if feature_dim == get_feature_spec("landmarks_plus_location_motion")["total_dim"]:
        return "landmarks_plus_location_motion"
    return "unknown"


def filter_records_by_feature_mode(records: list[dict], expected_mode: str) -> list[dict]:
    expected_total_dim = resolve_feature_spec(feature_mode=expected_mode)["total_dim"]
    compatible = []
    for record in records:
        if infer_record_feature_mode(record) != expected_mode:
            continue
        feature_dim = record.get("feature_dim")
        if feature_dim is not None and int(feature_dim) != expected_total_dim:
            continue
        compatible.append(record)
    return compatible
