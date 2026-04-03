from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.paths import ensure_dir


class DatasetWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        ensure_dir(output_dir)
        self.index_path = output_dir / "index.jsonl"

    def write_sample(
        self,
        sample_id: str,
        class_label: str,
        sequence: np.ndarray,
        metadata: dict,
        frame_valid_mask: np.ndarray | None = None,
        class_id: int | None = None,
    ) -> Path:
        sample_path = self.output_dir / f"{sample_id}.npz"
        payload = {
            "sequence": sequence.astype(np.float32),
            "class_label": np.array(class_label),
            "sample_id": np.array(sample_id),
            "metadata_json": np.array(json.dumps(metadata, ensure_ascii=False)),
        }
        if class_id is not None:
            payload["class_id"] = np.array([class_id], dtype=np.int32)
        if frame_valid_mask is not None:
            payload["frame_valid_mask"] = frame_valid_mask.astype(np.float32)
        np.savez_compressed(sample_path, **payload)
        index_record = {
            "sample_id": sample_id,
            "class_label": class_label,
            "path": sample_path.name,
            "sequence_length": int(sequence.shape[0]),
            "feature_dim": int(sequence.shape[1]),
            "feature_mode": metadata.get("feature_mode"),
            "metadata": metadata,
        }
        if class_id is not None:
            index_record["class_id"] = int(class_id)
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(index_record, ensure_ascii=False) + "\n")
        return sample_path
