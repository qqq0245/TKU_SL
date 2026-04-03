from __future__ import annotations

import csv
import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import ensure_dir


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build synthetic sentence videos by concatenating single-word clips.")
    parser.add_argument(
        "--source-root",
        default=str(PROJECT_ROOT / "datasets" / "training_variants" / "00_videos_training_30"),
        help="Folder containing the 30 single-word subfolders.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "datasets" / "sentences" / "00_sentence_videos_50"),
        help="Folder that will contain generated sentence videos.",
    )
    parser.add_argument(
        "--manifest-csv",
        default=str(PROJECT_ROOT / "metadata" / "sentence_video_manifest_50.csv"),
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="How many sentence videos to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sentence selection and clip sampling.",
    )
    parser.add_argument(
        "--gap-frames",
        type=int,
        default=2,
        help="How many repeated frames to insert between word clips.",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=480,
        help="Unified output video width.",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=320,
        help="Unified output video height.",
    )
    return parser


def iter_video_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def build_candidate_sentences() -> list[dict]:
    curated = [
        {"goal": "日常需求", "tokens": ["today", "i", "want", "food"], "grammar_note": "時間前置，需求句保持簡潔。"},
        {"goal": "日常需求", "tokens": ["today", "you", "want", "food"], "grammar_note": "時間前置，需求句保持簡潔。"},
        {"goal": "日常需求", "tokens": ["today", "student", "want", "book"], "grammar_note": "時間前置，需求句保持簡潔。"},
        {"goal": "日常需求", "tokens": ["tomorrow", "i", "want", "food"], "grammar_note": "時間前置，需求句保持簡潔。"},
        {"goal": "日常需求", "tokens": ["tomorrow", "you", "want", "book"], "grammar_note": "時間前置，需求句保持簡潔。"},
        {"goal": "日常需求", "tokens": ["tomorrow", "mother", "want", "food"], "grammar_note": "時間前置，需求句保持簡潔。"},
        {"goal": "詢問想吃什麼", "tokens": ["today", "you", "want", "eat", "what"], "grammar_note": "時間前置，疑問詞 what 置尾。"},
        {"goal": "詢問想吃什麼", "tokens": ["tomorrow", "you", "want", "eat", "what"], "grammar_note": "時間前置，疑問詞 what 置尾。"},
        {"goal": "詢問想吃什麼", "tokens": ["today", "i", "want", "eat", "what"], "grammar_note": "時間前置，疑問詞 what 置尾。"},
        {"goal": "詢問想吃什麼", "tokens": ["tomorrow", "student", "want", "eat", "what"], "grammar_note": "時間前置，疑問詞 what 置尾。"},
        {"goal": "詢問工作地點", "tokens": ["you", "work", "where"], "grammar_note": "WH 問句把 where 放在句尾。"},
        {"goal": "詢問工作地點", "tokens": ["i", "work", "where"], "grammar_note": "WH 問句把 where 放在句尾。"},
        {"goal": "詢問工作地點", "tokens": ["teacher", "work", "where"], "grammar_note": "WH 問句把 where 放在句尾。"},
        {"goal": "詢問工作地點", "tokens": ["student", "work", "where"], "grammar_note": "WH 問句把 where 放在句尾。"},
        {"goal": "詢問工作地點", "tokens": ["mother", "work", "where"], "grammar_note": "WH 問句把 where 放在句尾。"},
        {"goal": "詢問工作地點", "tokens": ["father", "work", "where"], "grammar_note": "WH 問句把 where 放在句尾。"},
        {"goal": "行動與方向", "tokens": ["mother", "go", "school", "help"], "grammar_note": "依動作順序排列，不補介系詞。"},
        {"goal": "行動與方向", "tokens": ["father", "go", "school", "help"], "grammar_note": "依動作順序排列，不補介系詞。"},
        {"goal": "行動與方向", "tokens": ["teacher", "go", "school", "help"], "grammar_note": "依動作順序排列，不補介系詞。"},
        {"goal": "行動與方向", "tokens": ["student", "go", "school", "help"], "grammar_note": "依動作順序排列，不補介系詞。"},
        {"goal": "行動與方向", "tokens": ["you", "go", "school", "help"], "grammar_note": "依動作順序排列，不補介系詞。"},
        {"goal": "行動與方向", "tokens": ["i", "go", "school", "help"], "grammar_note": "依動作順序排列，不補介系詞。"},
        {"goal": "喜好與特徵", "tokens": ["father", "like", "house", "beautiful"], "grammar_note": "形容詞放在名詞後，保留 ASL gloss 順序。"},
        {"goal": "喜好與特徵", "tokens": ["mother", "like", "house", "beautiful"], "grammar_note": "形容詞放在名詞後，保留 ASL gloss 順序。"},
        {"goal": "喜好與特徵", "tokens": ["you", "like", "book", "good"], "grammar_note": "形容詞放在名詞後，保留 ASL gloss 順序。"},
        {"goal": "喜好與特徵", "tokens": ["i", "like", "book", "good"], "grammar_note": "形容詞放在名詞後，保留 ASL gloss 順序。"},
        {"goal": "喜好與特徵", "tokens": ["you", "like", "dog", "good"], "grammar_note": "形容詞放在名詞後，保留 ASL gloss 順序。"},
        {"goal": "喜好與特徵", "tokens": ["student", "like", "book", "good"], "grammar_note": "形容詞放在名詞後，保留 ASL gloss 順序。"},
        {"goal": "喜好與特徵", "tokens": ["father", "like", "car", "good"], "grammar_note": "形容詞放在名詞後，保留 ASL gloss 順序。"},
        {"goal": "喜好與特徵", "tokens": ["mother", "like", "car", "good"], "grammar_note": "形容詞放在名詞後，保留 ASL gloss 順序。"},
        {"goal": "描述觀察", "tokens": ["i", "see", "dog", "bad"], "grammar_note": "觀察句保留主詞 + see + 名詞 + 形容詞。"},
        {"goal": "描述觀察", "tokens": ["you", "see", "dog", "bad"], "grammar_note": "觀察句保留主詞 + see + 名詞 + 形容詞。"},
        {"goal": "描述觀察", "tokens": ["teacher", "see", "dog", "good"], "grammar_note": "觀察句保留主詞 + see + 名詞 + 形容詞。"},
        {"goal": "描述觀察", "tokens": ["student", "see", "car", "bad"], "grammar_note": "觀察句保留主詞 + see + 名詞 + 形容詞。"},
        {"goal": "描述觀察", "tokens": ["mother", "see", "book", "good"], "grammar_note": "觀察句保留主詞 + see + 名詞 + 形容詞。"},
        {"goal": "描述觀察", "tokens": ["father", "see", "dog", "bad"], "grammar_note": "觀察句保留主詞 + see + 名詞 + 形容詞。"},
        {"goal": "描述狀態", "tokens": ["teacher", "happy"], "grammar_note": "狀態句省略 be 動詞。"},
        {"goal": "描述狀態", "tokens": ["student", "tired"], "grammar_note": "狀態句省略 be 動詞。"},
        {"goal": "描述狀態", "tokens": ["mother", "happy"], "grammar_note": "狀態句省略 be 動詞。"},
        {"goal": "描述狀態", "tokens": ["father", "tired"], "grammar_note": "狀態句省略 be 動詞。"},
        {"goal": "描述狀態", "tokens": ["you", "tired"], "grammar_note": "狀態句省略 be 動詞。"},
        {"goal": "描述狀態", "tokens": ["i", "happy"], "grammar_note": "狀態句省略 be 動詞。"},
        {"goal": "描述狀態", "tokens": ["teacher", "sad"], "grammar_note": "狀態句省略 be 動詞。"},
        {"goal": "描述狀態", "tokens": ["student", "happy"], "grammar_note": "狀態句省略 be 動詞。"},
        {"goal": "描述狀態", "tokens": ["mother", "sad"], "grammar_note": "狀態句省略 be 動詞。"},
        {"goal": "描述狀態", "tokens": ["father", "happy"], "grammar_note": "狀態句省略 be 動詞。"},
        {"goal": "雙子句狀態", "tokens": ["teacher", "happy", "student", "tired"], "grammar_note": "將兩個短子句直接串接，模擬連續表達。"},
        {"goal": "雙子句狀態", "tokens": ["mother", "happy", "father", "tired"], "grammar_note": "將兩個短子句直接串接，模擬連續表達。"},
        {"goal": "雙子句狀態", "tokens": ["teacher", "sad", "student", "happy"], "grammar_note": "將兩個短子句直接串接，模擬連續表達。"},
        {"goal": "雙子句狀態", "tokens": ["mother", "tired", "student", "happy"], "grammar_note": "將兩個短子句直接串接，模擬連續表達。"},
    ]

    if len(curated) != 50:
        raise ValueError(f"Expected 50 curated sentences, but got {len(curated)}.")
    return curated


def build_sentence_set(count: int, seed: int) -> list[dict]:
    candidates = build_candidate_sentences()
    if count > len(candidates):
        raise ValueError(f"Requested {count} sentences, but only {len(candidates)} unique candidates are available.")
    return candidates[:count]


def choose_source_clips(tokens: list[str], available: dict[str, list[Path]], rng: random.Random) -> list[Path]:
    chosen: list[Path] = []
    usage_counts: dict[str, dict[Path, int]] = {}
    for token in tokens:
        candidates = available.get(token, [])
        if not candidates:
            raise FileNotFoundError(f"No source videos found for token: {token}")
        per_word_usage = usage_counts.setdefault(token, {})
        least_used = min(per_word_usage.get(path, 0) for path in candidates) if per_word_usage else 0
        pool = [path for path in candidates if per_word_usage.get(path, 0) == least_used]
        clip_path = rng.choice(pool)
        per_word_usage[clip_path] = per_word_usage.get(clip_path, 0) + 1
        chosen.append(clip_path)
    return chosen


def read_video_frames(video_path: Path, target_size: tuple[int, int] | None) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolved_size = target_size or (width, height)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != resolved_size[0] or frame.shape[0] != resolved_size[1]:
            frame = cv2.resize(frame, resolved_size, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    return frames, fps, resolved_size


def concat_sentence_video(
    source_paths: list[Path],
    output_path: Path,
    gap_frames: int,
    output_size: tuple[int, int],
) -> dict:
    ensure_dir(output_path.parent)
    all_frames: list[np.ndarray] = []
    fps_values: list[float] = []
    source_frame_counts: list[int] = []

    for clip_index, source_path in enumerate(source_paths):
        frames, fps, _ = read_video_frames(source_path, output_size)
        fps_values.append(fps)
        source_frame_counts.append(len(frames))
        all_frames.extend(frames)
        if clip_index < len(source_paths) - 1 and gap_frames > 0:
            all_frames.extend([frames[-1].copy() for _ in range(gap_frames)])

    if not all_frames:
        raise RuntimeError(f"No frames collected for output: {output_path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(round(sum(fps_values) / len(fps_values), 2)),
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open writer for output: {output_path}")
    for frame in all_frames:
        writer.write(frame)
    writer.release()
    return {
        "fps": float(round(sum(fps_values) / len(fps_values), 2)),
        "frame_count": len(all_frames),
        "width": output_size[0],
        "height": output_size[1],
        "source_frame_counts": source_frame_counts,
    }


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest_csv)
    rng = random.Random(args.seed)
    output_size = (args.output_width, args.output_height)

    available = {folder.name: iter_video_files(folder) for folder in sorted(source_root.iterdir()) if folder.is_dir()}
    sentences = build_sentence_set(args.count, args.seed)
    ensure_dir(output_root)
    ensure_dir(manifest_path.parent)

    manifest_rows: list[dict] = []
    for index, sentence in enumerate(sentences, start=1):
        sentence_id = f"sentence_{index:03d}"
        tokens = sentence["tokens"]
        gloss = " ".join(tokens).upper()
        chosen_sources = choose_source_clips(tokens, available, rng)
        file_slug = "_".join(tokens)
        output_path = output_root / f"{sentence_id}__{file_slug}.mp4"
        video_stats = concat_sentence_video(chosen_sources, output_path, args.gap_frames, output_size)
        manifest_rows.append(
            {
                "sentence_id": sentence_id,
                "goal": sentence["goal"],
                "gloss": gloss,
                "token_count": len(tokens),
                "tokens_json": json.dumps(tokens, ensure_ascii=False),
                "source_clips_json": json.dumps([str(path.resolve()) for path in chosen_sources], ensure_ascii=False),
                "output_video": str(output_path.resolve()),
                "fps": video_stats["fps"],
                "frame_count": video_stats["frame_count"],
                "width": video_stats["width"],
                "height": video_stats["height"],
                "source_frame_counts_json": json.dumps(video_stats["source_frame_counts"]),
                "grammar_note": sentence["grammar_note"],
            }
        )

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sentence_id",
                "goal",
                "gloss",
                "token_count",
                "tokens_json",
                "source_clips_json",
                "output_video",
                "fps",
                "frame_count",
                "width",
                "height",
                "source_frame_counts_json",
                "grammar_note",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print(f"Generated videos: {len(manifest_rows)}")
    print(f"Manifest CSV: {manifest_path}")


if __name__ == "__main__":
    main()
