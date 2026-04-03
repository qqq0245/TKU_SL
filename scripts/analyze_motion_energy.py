from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports" / "realtime_tests"


def find_latest_frame_predictions_csv(root: Path) -> Path:
    candidates = sorted(root.glob("realtime_test_*/frame_predictions.csv"))
    if not candidates:
        raise FileNotFoundError(f"No frame_predictions.csv found under: {root}")
    # Folder names include timestamps, so lexicographic sort gives latest at the end.
    return candidates[-1]


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile on empty list.")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)

    ordered = sorted(values)
    rank = (len(ordered) - 1) * (q / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def iter_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def collect_idle_motion_energy(csv_path: Path) -> list[float]:
    values: list[float] = []
    for row in iter_rows(csv_path):
        raw_label = (row.get("raw_label") or "").strip().lower()
        status = (row.get("status") or "").strip().lower()
        if raw_label == "no_sign" or status == "trigger_idle":
            try:
                values.append(float(row["motion_energy"]))
            except (KeyError, TypeError, ValueError):
                continue
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze idle-frame motion_energy baseline from frame_predictions.csv")
    parser.add_argument(
        "--csv",
        default="",
        help="Optional path to frame_predictions.csv. If omitted, auto-picks latest under reports/realtime_tests/.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve() if args.csv else find_latest_frame_predictions_csv(REPORTS_ROOT)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    idle_values = collect_idle_motion_energy(csv_path)
    if not idle_values:
        raise RuntimeError("No idle frames found. Check raw_label/status filters and CSV content.")

    mean_value = sum(idle_values) / len(idle_values)
    p90_value = percentile(idle_values, 90.0)
    max_value = max(idle_values)

    print(f"CSV: {csv_path}")
    print(f"Idle frames: {len(idle_values)}")
    print(f"Mean: {mean_value:.6f}")
    print(f"P90: {p90_value:.6f}")
    print(f"Max: {max_value:.6f}")


if __name__ == "__main__":
    main()
