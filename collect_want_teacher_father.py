from __future__ import annotations

"""
want / teacher / father 最小增量補資料腳本
==========================================

用途：
1. 用 webcam 直接錄製 `want` / `teacher` / `father`
2. 使用既有資料路徑，維持 `pose_local_anchor=torso_center`
3. 只把這三類新影片增量加入現有 training set
4. 只對這三類做 landmarks / sequence 匯出
5. 重建 splits，接著執行最小 retrain 指令

中文操作步驟：
1. 執行本檔：`python collect_want_teacher_father.py`
2. 依序錄製：
   - `want / 想`
   - `teacher / 老師`
   - `father / 父親`
3. 每個詞預設錄 `15` 段，每段 `3` 秒，總長約 `45` 秒
4. 每段開始前會自動倒數，請只做單一詞的 exact-span 動作，不要混入過渡手勢
5. 每段錄影時請讓上半身、雙手與臉都完整出現在畫面中
6. 錄完三個詞後，腳本會自動：
   - 複製新影片到既有 training variant root
   - 更新 `source_metadata.csv`
   - 重掃 manifest
   - 抽 landmarks
   - 匯出 sequences
   - 重建 train/val/test splits
   - 啟動 retrain

錄製建議：
- 背景單純、光線穩定
- 每段只做一個詞，不要連續接別的詞
- 起手前與收手後各保留一點靜止幀
- 若畫面不穩或手被切掉，直接 `Ctrl+C` 中止後重跑
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT / "integration_workspace"
PIPELINE_ROOT = WORKSPACE_ROOT / "dataset_pipeline_webcam9_relative_coord_v1"
ARTIFACT_ROOT = WORKSPACE_ROOT / "artifacts_webcam9_relative_coord_v1"
RECORDED_ROOT = PROJECT_ROOT / "datasets" / "recorded" / "webcam_30_words"
TRAINING_VARIANT_ROOT = PROJECT_ROOT / "datasets" / "training_variants" / "webcam_9_with_nosign_raw"
SOURCE_METADATA_CSV = TRAINING_VARIANT_ROOT / "source_metadata.csv"

VIDEO_MANIFEST_CSV = PIPELINE_ROOT / "manifests" / "video_manifest.csv"
SEQUENCE_MANIFEST_CSV = PIPELINE_ROOT / "manifests" / "sequence_manifest.csv"
LABEL_MAP_JSON = PIPELINE_ROOT / "manifests" / "label_map.json"
INDEX_JSONL = PIPELINE_ROOT / "processed_sequences" / "index.jsonl"
TRAIN_SPLIT_CSV = PIPELINE_ROOT / "splits" / "train.csv"
VAL_SPLIT_CSV = PIPELINE_ROOT / "splits" / "val.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

POSE_LOCAL_ANCHOR = "torso_center"
WINDOW_NAME = "Collect want / teacher / father"
TARGET_TOKENS = [
    ("want", "想"),
    ("teacher", "老師"),
    ("father", "父親"),
]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}


@dataclass
class CollectionConfig:
    camera_index: int
    clips_per_token: int
    seconds_per_clip: float
    countdown_seconds: int
    width: int
    height: int
    fps: float


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Collect want/teacher/father webcam clips and run incremental retraining.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--clips-per-token", type=int, default=15)
    parser.add_argument("--seconds-per-clip", type=float, default=3.0)
    parser.add_argument("--countdown-seconds", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_label_map() -> dict:
    with LABEL_MAP_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_intro(config: CollectionConfig) -> None:
    total_seconds = config.clips_per_token * config.seconds_per_clip
    print("=" * 72)
    print("want / teacher / father 最小資料補強流程")
    print("=" * 72)
    print(f"pose_local_anchor: {POSE_LOCAL_ANCHOR}")
    print(f"每個詞預設總時長: {total_seconds:.0f} 秒")
    print(f"每個詞段數: {config.clips_per_token}")
    print(f"每段秒數: {config.seconds_per_clip:.1f}")
    print("")
    print("操作步驟：")
    print("1. 畫面要包含上半身、雙手、臉。")
    print("2. 每一段只做單一詞的 exact-span 動作。")
    print("3. 每段開始前會倒數，倒數結束後開始錄影。")
    print("4. 錄影途中按 Q 或關閉視窗可中止。")
    print("5. 三個詞錄完後，腳本會自動跑增量抽特徵與 retrain。")
    print("")


def open_camera(camera_index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"無法開啟 webcam，camera_index={camera_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def draw_overlay(frame, lines: list[str], color: tuple[int, int, int] = (0, 255, 0)):
    rendered = frame.copy()
    top = 32
    for line in lines:
        cv2.putText(rendered, line, (24, top), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(rendered, line, (24, top), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        top += 34
    return rendered


def wait_or_abort(delay_ms: int = 1) -> None:
    key = cv2.waitKey(delay_ms) & 0xFF
    if key == ord("q"):
        raise KeyboardInterrupt("使用者中止錄影。")


def record_clip(
    capture: cv2.VideoCapture,
    output_path: Path,
    token: str,
    zh_label: str,
    clip_index: int,
    total_clips: int,
    config: CollectionConfig,
) -> None:
    ensure_dir(output_path.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, config.fps, (config.width, config.height))
    if not writer.isOpened():
        raise RuntimeError(f"無法建立輸出影片：{output_path}")

    try:
        for remaining in range(config.countdown_seconds, 0, -1):
            frame_start = time.time()
            while time.time() - frame_start < 1.0:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("webcam 讀取失敗。")
                frame = cv2.resize(frame, (config.width, config.height))
                shown = draw_overlay(
                    frame,
                    [
                        f"Token: {token} ({zh_label})",
                        f"Clip: {clip_index}/{total_clips}",
                        f"Recording starts in {remaining}",
                        "Press Q to abort",
                    ],
                    color=(0, 255, 255),
                )
                cv2.imshow(WINDOW_NAME, shown)
                wait_or_abort(1)

        clip_start = time.time()
        while time.time() - clip_start < config.seconds_per_clip:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("webcam 讀取失敗。")
            frame = cv2.resize(frame, (config.width, config.height))
            writer.write(frame)
            remaining = max(0.0, config.seconds_per_clip - (time.time() - clip_start))
            shown = draw_overlay(
                frame,
                [
                    f"Token: {token} ({zh_label})",
                    f"Clip: {clip_index}/{total_clips}",
                    f"REC {remaining:0.1f}s",
                    "Keep only the exact-span sign",
                ],
                color=(0, 0, 255),
            )
            cv2.imshow(WINDOW_NAME, shown)
            wait_or_abort(1)
    finally:
        writer.release()


def collect_token_clips(capture: cv2.VideoCapture, token: str, zh_label: str, config: CollectionConfig) -> list[Path]:
    target_dir = RECORDED_ROOT / token
    ensure_dir(target_dir)
    print(f"\n準備錄製 {token} / {zh_label}")
    print("請按 Enter 開始；若要中止整個流程，直接 Ctrl+C。")
    input()

    collected: list[Path] = []
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for clip_index in range(1, config.clips_per_token + 1):
        clip_name = f"{token}_{session_stamp}_{clip_index:03d}.mp4"
        output_path = target_dir / clip_name
        record_clip(capture, output_path, token, zh_label, clip_index, config.clips_per_token, config)
        collected.append(output_path)
        print(f"[錄製] {output_path}")
    return collected


def iter_token_videos(token: str):
    token_dir = RECORDED_ROOT / token
    if not token_dir.exists():
        return []
    return sorted(
        path
        for path in token_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def load_source_metadata_rows() -> tuple[list[dict], set[str]]:
    if not SOURCE_METADATA_CSV.exists():
        return [], set()
    with SOURCE_METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {row["video_path"] for row in rows if row.get("video_path")}


def sync_recorded_targets_to_training_variant() -> dict[str, int]:
    ensure_dir(TRAINING_VARIANT_ROOT)
    existing_rows, existing_video_paths = load_source_metadata_rows()
    appended_rows = []
    copied_count = 0
    discovered_count = 0

    for token, _ in TARGET_TOKENS:
        src_dir = RECORDED_ROOT / token
        dst_dir = TRAINING_VARIANT_ROOT / token
        ensure_dir(dst_dir)
        for src_path in iter_token_videos(token):
            discovered_count += 1
            dst_path = dst_dir / src_path.name
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            dst_resolved = str(dst_path.resolve())
            if dst_resolved not in existing_video_paths:
                appended_rows.append(
                    {
                        "video_path": dst_resolved,
                        "source_origin_path": str(src_path.resolve()),
                        "source_group_key": str(src_path.resolve()),
                    }
                )
                existing_video_paths.add(dst_resolved)

    all_rows = existing_rows + appended_rows
    with SOURCE_METADATA_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "source_origin_path", "source_group_key"])
        writer.writeheader()
        writer.writerows(all_rows)

    return {
        "recorded_target_video_count": discovered_count,
        "copied_to_training_variant_count": copied_count,
        "metadata_rows_added_count": len(appended_rows),
    }


def rebuild_full_sequence_manifest() -> int:
    rows = []
    if not INDEX_JSONL.exists():
        raise RuntimeError(f"找不到 index.jsonl：{INDEX_JSONL}")
    with INDEX_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            sample_path = (INDEX_JSONL.parent / record["path"]).resolve()
            rows.append(
                {
                    "sample_id": record.get("sample_id", metadata.get("sample_id", "")),
                    "sample_path": str(sample_path),
                    "class_id": record.get("class_id", metadata.get("class_id", "")),
                    "english_label": record.get("class_label", metadata.get("label", "")),
                    "zh_tw_translation": metadata.get("zh_tw_translation", ""),
                    "feature_mode": record.get("feature_mode") or metadata.get("feature_mode", ""),
                    "feature_dim": record.get("feature_dim", metadata.get("feature_dim", "")),
                    "sequence_length": record.get("sequence_length", metadata.get("sequence_length", "")),
                    "source_video_path": metadata.get("source_video_path", ""),
                    "source_origin_path": metadata.get("source_origin_path", ""),
                    "source_group_key": metadata.get("source_group_key", ""),
                }
            )

    ensure_dir(SEQUENCE_MANIFEST_CSV.parent)
    with SEQUENCE_MANIFEST_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "sample_path",
                "class_id",
                "english_label",
                "zh_tw_translation",
                "feature_mode",
                "feature_dim",
                "sequence_length",
                "source_video_path",
                "source_origin_path",
                "source_group_key",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_session_summary(config: CollectionConfig, sync_stats: dict[str, int], sequence_manifest_rows: int) -> Path:
    ensure_dir(REPORT_DIR)
    output_path = REPORT_DIR / "want_teacher_father_collection_last_run.json"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "targets": [token for token, _ in TARGET_TOKENS],
        "pose_local_anchor": POSE_LOCAL_ANCHOR,
        "clips_per_token": config.clips_per_token,
        "seconds_per_clip": config.seconds_per_clip,
        "training_variant_root": str(TRAINING_VARIANT_ROOT.resolve()),
        "dataset_pipeline_root": str(PIPELINE_ROOT.resolve()),
        "artifacts_root": str(ARTIFACT_ROOT.resolve()),
        "sync_stats": sync_stats,
        "full_sequence_manifest_rows": sequence_manifest_rows,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    rendered = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"[CMD] {rendered}")
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def run_incremental_pipeline() -> None:
    env = os.environ.copy()
    env["SIGN_DATASET_PIPELINE_ROOT"] = str(PIPELINE_ROOT)
    env["SIGN_ARTIFACTS_DIR"] = str(ARTIFACT_ROOT)

    class_filter = [token for token, _ in TARGET_TOKENS]
    run_command(
        [
            sys.executable,
            str(WORKSPACE_ROOT / "scripts" / "scan_video_dataset.py"),
            "--source-root",
            str(TRAINING_VARIANT_ROOT),
            "--label-map-json",
            str(LABEL_MAP_JSON),
        ],
        cwd=WORKSPACE_ROOT,
        env=env,
    )
    run_command(
        [
            sys.executable,
            str(WORKSPACE_ROOT / "scripts" / "extract_landmarks_batch.py"),
            "--manifest-csv",
            str(VIDEO_MANIFEST_CSV),
            "--class-filter",
            *class_filter,
            "--skip-existing",
        ],
        cwd=WORKSPACE_ROOT,
        env=env,
    )
    run_command(
        [
            sys.executable,
            str(WORKSPACE_ROOT / "scripts" / "export_sequences_batch.py"),
            "--manifest-csv",
            str(VIDEO_MANIFEST_CSV),
            "--sequence-length",
            "30",
            "--stride",
            "5",
            "--class-filter",
            *class_filter,
            "--skip-existing",
        ],
        cwd=WORKSPACE_ROOT,
        env=env,
    )
    run_command(
        [
            sys.executable,
            str(WORKSPACE_ROOT / "scripts" / "build_splits.py"),
            "--index-jsonl",
            str(INDEX_JSONL),
            "--output-dir",
            str(PIPELINE_ROOT / "splits"),
        ],
        cwd=WORKSPACE_ROOT,
        env=env,
    )
    run_command(
        [
            sys.executable,
            str(WORKSPACE_ROOT / "train_multibranch.py"),
            "--data-dir",
            str(PIPELINE_ROOT / "processed_sequences"),
            "--train-split-csv",
            str(TRAIN_SPLIT_CSV),
            "--val-split-csv",
            str(VAL_SPLIT_CSV),
        ],
        cwd=WORKSPACE_ROOT,
        env=env,
    )


def validate_targets_exist(label_map: dict) -> None:
    by_label = label_map.get("by_english_label", {})
    missing = [token for token, _ in TARGET_TOKENS if token not in by_label]
    if missing:
        raise RuntimeError(f"label_map.json 缺少目標類別：{missing}")


def main() -> None:
    args = build_parser().parse_args()
    config = CollectionConfig(
        camera_index=args.camera_index,
        clips_per_token=args.clips_per_token,
        seconds_per_clip=args.seconds_per_clip,
        countdown_seconds=args.countdown_seconds,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    label_map = load_label_map()
    validate_targets_exist(label_map)
    print_intro(config)

    capture = None
    try:
        capture = open_camera(config.camera_index, config.width, config.height, config.fps)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, config.width, config.height)
        for token, zh_label in TARGET_TOKENS:
            collect_token_clips(capture, token, zh_label, config)
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()

    sync_stats = sync_recorded_targets_to_training_variant()
    run_incremental_pipeline()
    sequence_manifest_rows = rebuild_full_sequence_manifest()
    summary_path = write_session_summary(config, sync_stats, sequence_manifest_rows)

    print("")
    print("=" * 72)
    print("增量補資料與 retrain 已串接完成")
    print(f"pose_local_anchor: {POSE_LOCAL_ANCHOR}")
    print(f"summary: {summary_path}")
    print("最小 retrain 指令：")
    print(f"python {PROJECT_ROOT / 'collect_want_teacher_father.py'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
