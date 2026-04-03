from __future__ import annotations

import json
from pathlib import Path


def label_registry_path(processed_dir: Path) -> Path:
    return processed_dir / "labels.json"


def load_label_registry(processed_dir: Path) -> list[str]:
    path = label_registry_path(processed_dir)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_label_registered(processed_dir: Path, label: str) -> None:
    labels = load_label_registry(processed_dir)
    if label not in labels:
        labels.append(label)
        labels = sorted(labels)
        with label_registry_path(processed_dir).open("w", encoding="utf-8") as handle:
            json.dump(labels, handle, ensure_ascii=False, indent=2)
