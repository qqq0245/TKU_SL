from __future__ import annotations

import csv
import json
import statistics
import sys
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.scripted_video_alignment import parse_script_tokens_from_path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Derive activity-based token span hypotheses from frame_predictions.csv.")
    parser.add_argument("--video", required=True, help="Source video path.")
    parser.add_argument("--frame-csv", required=True, help="frame_predictions.csv path.")
    parser.add_argument(
        "--segments-manifest",
        default=str(PROJECT_ROOT / "metadata" / "sentence_word_segments_manifest_50.csv"),
        help="Word segment manifest used to estimate expected per-token durations.",
    )
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    parser.add_argument("--min-motion-energy", type=float, default=0.04)
    parser.add_argument("--min-signal-score", type=float, default=0.50)
    parser.add_argument("--gap-frames", type=int, default=15)
    parser.add_argument("--min-chunk-frames", type=int, default=5)
    parser.add_argument("--gap-penalty-scale", type=float, default=25.0)
    return parser


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_expected_lengths(manifest_path: Path, tokens: list[str]) -> dict[str, float]:
    rows = _read_csv_rows(manifest_path)
    values: dict[str, list[int]] = {token: [] for token in tokens}
    for row in rows:
        token = row["token"].strip().lower()
        if token in values:
            values[token].append(int(row["frame_count"]))
    expected: dict[str, float] = {}
    for token in tokens:
        samples = values.get(token, [])
        if samples:
            expected[token] = float(statistics.median(samples))
        else:
            expected[token] = 60.0
    return expected


def _detect_activity_chunks(
    rows: list[dict[str, str]],
    *,
    min_motion_energy: float,
    min_signal_score: float,
    gap_frames: int,
) -> list[dict[str, float | int]]:
    chunks: list[dict[str, float | int]] = []
    active_start: int | None = None
    last_active: int | None = None
    gap = 0

    for row in rows:
        frame_index = int(row["frame_index"])
        motion_energy = float(row["motion_energy"])
        signal_score = float(row["signal_score"])
        has_hand = bool(int(row["left_hand_present"]) or int(row["right_hand_present"]))
        active = has_hand or motion_energy >= min_motion_energy or signal_score >= min_signal_score
        if active:
            if active_start is None:
                active_start = frame_index
            last_active = frame_index
            gap = 0
            continue
        if active_start is None:
            continue
        gap += 1
        if gap < gap_frames:
            continue
        span_rows = rows[active_start : last_active + 1]
        chunks.append(
            {
                "start_frame": active_start,
                "end_frame": last_active,
                "frame_count": int(last_active - active_start + 1),
                "mean_motion_energy": sum(float(item["motion_energy"]) for item in span_rows) / max(len(span_rows), 1),
                "peak_motion_energy": max(float(item["motion_energy"]) for item in span_rows),
            }
        )
        active_start = None
        last_active = None
        gap = 0

    if active_start is not None and last_active is not None:
        span_rows = rows[active_start : last_active + 1]
        chunks.append(
            {
                "start_frame": active_start,
                "end_frame": last_active,
                "frame_count": int(last_active - active_start + 1),
                "mean_motion_energy": sum(float(item["motion_energy"]) for item in span_rows) / max(len(span_rows), 1),
                "peak_motion_energy": max(float(item["motion_energy"]) for item in span_rows),
            }
        )
    return chunks


def _assign_chunks_to_tokens(
    chunks: list[dict[str, float | int]],
    tokens: list[str],
    expected_lengths: dict[str, float],
    gap_penalty_scale: float,
) -> tuple[float, list[dict[str, object]]]:
    if not chunks or not tokens:
        return 0.0, []

    prefix_lengths = [0]
    for chunk in chunks:
        prefix_lengths.append(prefix_lengths[-1] + int(chunk["frame_count"]))

    @lru_cache(maxsize=None)
    def solve(token_index: int, chunk_index: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if token_index == len(tokens) and chunk_index == len(chunks):
            return 0.0, tuple()
        if token_index == len(tokens) or chunk_index == len(chunks):
            return float("inf"), tuple()

        best_cost = float("inf")
        best_ranges: tuple[tuple[int, int], ...] = tuple()
        max_take = len(chunks) - chunk_index - (len(tokens) - token_index - 1)
        for take in range(1, max_take + 1):
            start_idx = chunk_index
            end_idx = chunk_index + take - 1
            total_frames = prefix_lengths[end_idx + 1] - prefix_lengths[start_idx]
            expected = max(expected_lengths.get(tokens[token_index], 60.0), 1.0)
            duration_cost = ((float(total_frames) - expected) / expected) ** 2
            gap_cost = 0.0
            for idx in range(start_idx, end_idx):
                gap_frames = max(0, int(chunks[idx + 1]["start_frame"]) - int(chunks[idx]["end_frame"]) - 1)
                gap_cost += float(gap_frames) / max(gap_penalty_scale, 1.0)
            rest_cost, rest_ranges = solve(token_index + 1, chunk_index + take)
            score = duration_cost + gap_cost + rest_cost
            if score < best_cost:
                best_cost = score
                best_ranges = ((start_idx, end_idx),) + rest_ranges
        return best_cost, best_ranges

    total_cost, chunk_ranges = solve(0, 0)
    assignments: list[dict[str, object]] = []
    for token, (start_idx, end_idx) in zip(tokens, chunk_ranges):
        start_frame = int(chunks[start_idx]["start_frame"])
        end_frame = int(chunks[end_idx]["end_frame"])
        active_frame_count = int(sum(int(chunks[idx]["frame_count"]) for idx in range(start_idx, end_idx + 1)))
        assignments.append(
            {
                "token": token,
                "chunk_start_index": int(start_idx),
                "chunk_end_index": int(end_idx),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "active_frame_count": active_frame_count,
                "expected_frame_count_median": expected_lengths.get(token, 60.0),
                "merged_chunk_lengths": [int(chunks[idx]["frame_count"]) for idx in range(start_idx, end_idx + 1)],
            }
        )
    return total_cost, assignments


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.video).resolve()
    frame_csv_path = Path(args.frame_csv).resolve()
    output_json = Path(args.output_json).resolve()
    manifest_path = Path(args.segments_manifest).resolve()

    tokens = parse_script_tokens_from_path(video_path)
    frame_rows = _read_csv_rows(frame_csv_path)
    detected_chunks = _detect_activity_chunks(
        frame_rows,
        min_motion_energy=float(args.min_motion_energy),
        min_signal_score=float(args.min_signal_score),
        gap_frames=int(args.gap_frames),
    )
    significant_chunks = [chunk for chunk in detected_chunks if int(chunk["frame_count"]) >= int(args.min_chunk_frames)]
    dropped_chunks = [chunk for chunk in detected_chunks if int(chunk["frame_count"]) < int(args.min_chunk_frames)]
    expected_lengths = _load_expected_lengths(manifest_path, tokens)
    assignment_cost, assignments = _assign_chunks_to_tokens(
        significant_chunks,
        tokens,
        expected_lengths,
        float(args.gap_penalty_scale),
    )

    payload = {
        "video_path": str(video_path),
        "frame_csv_path": str(frame_csv_path),
        "tokens": tokens,
        "expected_lengths": expected_lengths,
        "detected_chunks": detected_chunks,
        "significant_chunks": significant_chunks,
        "dropped_chunks": dropped_chunks,
        "assignment_cost": assignment_cost,
        "token_span_hypothesis": assignments,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_json}")
    print(json.dumps({"tokens": tokens, "assignment_cost": assignment_cost}, ensure_ascii=False))
    for row in assignments:
        print(
            f"{row['token']}: {row['start_frame']}-{row['end_frame']} "
            f"active={row['active_frame_count']} expected={row['expected_frame_count_median']}"
        )


if __name__ == "__main__":
    main()
