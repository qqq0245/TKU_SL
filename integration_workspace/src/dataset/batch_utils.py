from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from src.utils.paths import ensure_dir


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(
    rows: list[dict],
    *,
    class_filter: list[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list[dict]:
    filtered = rows
    if class_filter:
        target = {item.strip() for item in class_filter if item.strip()}
        filtered = [row for row in filtered if row.get("english_label") in target or str(row.get("class_id")) in target]
    if offset:
        filtered = filtered[offset:]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def merge_failure_rows(
    existing_rows: list[dict],
    new_failures: list[dict],
    *,
    key_field: str,
    resolved_keys: set[str] | None = None,
) -> list[dict]:
    resolved_keys = resolved_keys or set()
    merged: dict[str, dict] = {}
    for row in existing_rows:
        key = row.get(key_field)
        if key and key not in resolved_keys:
            merged[key] = row
    for row in new_failures:
        key = row.get(key_field)
        if key:
            merged[key] = row
    return list(merged.values())


def group_count(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] += 1
    return dict(counts)


__all__ = [
    "filter_rows",
    "group_count",
    "merge_failure_rows",
    "read_csv_rows",
    "write_csv_rows",
]
