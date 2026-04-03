from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import CONFIG


DEFAULT_EXTRA_LABELS = ["no_sign", "transition"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msjh.ttc"),
    Path(r"C:\Windows\Fonts\msjhbd.ttc"),
    Path(r"C:\Windows\Fonts\NotoSansTC-VF.ttf"),
    Path(r"C:\Windows\Fonts\mingliu.ttc"),
)


@dataclass
class RecordingConfig:
    output_root: Path
    manifest_csv: Path
    camera_index: int = 0
    fps: float = 20.0
    width: int = 960
    height: int = 540
    countdown_seconds: int = 3
    clip_seconds: float = 2.4
    segment_gap_seconds: float = 1.0
    mirror_preview: bool = True


def load_default_labels() -> list[str]:
    vocabulary_csv = CONFIG.project_root.parent / "metadata" / "training_30_vocabulary.csv"
    labels: list[str] = []
    if vocabulary_csv.exists():
        with vocabulary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                label = str(row.get("english_label", "")).strip().lower()
                if label and label not in labels:
                    labels.append(label)
    for extra in DEFAULT_EXTRA_LABELS:
        if extra not in labels:
            labels.append(extra)
    return labels


def ensure_output_structure(output_root: Path, labels: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (output_root / label).mkdir(parents=True, exist_ok=True)


def append_manifest_row(manifest_csv: Path, row: dict[str, str]) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "label",
        "file_path",
        "camera_index",
        "fps",
        "frame_count",
        "width",
        "height",
        "clip_seconds",
        "recorded_at",
    ]
    file_exists = manifest_csv.exists()
    with manifest_csv.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def refresh_label_summary(manifest_csv: Path, summary_csv: Path) -> None:
    if not manifest_csv.exists():
        return

    grouped: dict[str, dict[str, str | int | float]] = {}
    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("label", "")).strip().lower()
            if not label:
                continue
            clip_seconds = float(row.get("clip_seconds", "0") or 0)
            frame_count = int(float(row.get("frame_count", "0") or 0))
            label_summary = grouped.setdefault(
                label,
                {
                    "label": label,
                    "clip_count": 0,
                    "total_seconds": 0.0,
                    "total_frames": 0,
                    "last_recorded_at": "",
                },
            )
            label_summary["clip_count"] = int(label_summary["clip_count"]) + 1
            label_summary["total_seconds"] = float(label_summary["total_seconds"]) + clip_seconds
            label_summary["total_frames"] = int(label_summary["total_frames"]) + frame_count
            label_summary["last_recorded_at"] = str(row.get("recorded_at", "")).strip()

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "clip_count", "total_seconds", "total_frames", "last_recorded_at"],
        )
        writer.writeheader()
        for label in sorted(grouped):
            row = grouped[label]
            writer.writerow(
                {
                    "label": row["label"],
                    "clip_count": row["clip_count"],
                    "total_seconds": f"{float(row['total_seconds']):.2f}",
                    "total_frames": row["total_frames"],
                    "last_recorded_at": row["last_recorded_at"],
                }
            )


def _iter_existing_indices(label_dir: Path) -> int:
    max_index = 0
    for path in label_dir.iterdir():
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            stem_parts = path.stem.split("_")
            for part in reversed(stem_parts):
                if part.isdigit():
                    max_index = max(max_index, int(part))
                    break
    return max_index


def _next_clip_path(output_root: Path, label: str) -> tuple[Path, str]:
    label_dir = output_root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    next_index = _iter_existing_indices(label_dir) + 1
    clip_id = f"{label}_{timestamp}_{next_index:03d}"
    return label_dir / f"{clip_id}.mp4", clip_id


def _resolve_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _save_recorded_clip(
    config: RecordingConfig,
    label: str,
    frames: list,
    frame_size: tuple[int, int] | None,
) -> str | None:
    if frame_size is None or len(frames) < 2:
        return None

    output_path, clip_id = _next_clip_path(config.output_root, label)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.fps),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"無法寫入影片：{output_path}")
    for saved_frame in frames:
        writer.write(saved_frame)
    writer.release()

    clip_seconds = len(frames) / max(config.fps, 1.0)
    append_manifest_row(
        config.manifest_csv,
        {
            "clip_id": clip_id,
            "label": label,
            "file_path": str(output_path.resolve()),
            "camera_index": str(config.camera_index),
            "fps": f"{config.fps:.2f}",
            "frame_count": str(len(frames)),
            "width": str(frame_size[0]),
            "height": str(frame_size[1]),
            "clip_seconds": f"{clip_seconds:.2f}",
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    refresh_label_summary(
        config.manifest_csv,
        config.manifest_csv.with_name(f"{config.manifest_csv.stem}_label_summary.csv"),
    )
    return clip_id


def _draw_overlay(
    frame,
    *,
    label: str,
    label_index: int,
    total_labels: int,
    saved_count: int,
    status_line: str,
    hint_line: str,
) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    title_font = _resolve_font(28)
    body_font = _resolve_font(22)

    lines = [
        (f"Label {label_index + 1}/{total_labels}: {label}", (60, 255, 60), title_font),
        (f"Saved clips: {saved_count}", (255, 255, 255), body_font),
        (status_line, (255, 220, 120), body_font),
        (hint_line, (220, 220, 220), body_font),
        ("[Space] 開始/暫停  [N] 下一個  [P] 上一個  [Q] 離開", (220, 220, 220), body_font),
    ]

    y = 20
    for text, color, font in lines:
        draw.text((20, y), text, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(0, 0, 0))
        draw.text((20, y), text, font=font, fill=color)
        y += 38

    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _label_specific_hint(label: str) -> str:
    normalized = label.strip().lower()
    if normalized == "no_sign":
        return "請自然待機、移動或調整姿勢，但不要做手語。"
    if normalized == "transition":
        return "請錄起手、收手、半段動作或詞與詞之間的過渡。"
    return "請做完整單字，維持上半身與雙手都在畫面內。"


def run_webcam_data_recorder(config: RecordingConfig, labels: list[str] | None = None) -> None:
    labels = labels or load_default_labels()
    ensure_output_structure(config.output_root, labels)

    capture = cv2.VideoCapture(config.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"無法開啟 webcam，camera_index={config.camera_index}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    capture.set(cv2.CAP_PROP_FPS, config.fps)

    label_index = 0
    window_name = "Webcam Data Recorder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    frame_count_target = max(2, int(round(config.clip_seconds * config.fps)))
    gap_frame_target = max(0, int(round(config.segment_gap_seconds * config.fps)))
    is_recording = False
    cooldown_frames_remaining = 0
    recorded_frames: list = []
    frame_size: tuple[int, int] | None = None
    segment_counter = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("無法從 webcam 讀取畫面。")
            if config.mirror_preview:
                frame = cv2.flip(frame, 1)

            label = labels[label_index]
            saved_count = len(list((config.output_root / label).glob("*.mp4")))
            preview = frame.copy()
            if is_recording and cooldown_frames_remaining > 0:
                status_line = f"間隔中 {cooldown_frames_remaining}/{gap_frame_target}"
                hint_line = "1 秒後會自動開始下一段，按空白鍵可暫停。"
                cooldown_frames_remaining -= 1
            elif is_recording:
                if frame_size is None:
                    frame_size = (int(frame.shape[1]), int(frame.shape[0]))
                recorded_frames.append(frame.copy())
                status_line = f"連續錄製中 第 {segment_counter + 1} 段 {len(recorded_frames)}/{frame_count_target}"
                hint_line = "按空白鍵暫停，系統會自動每段存檔。"
                if len(recorded_frames) >= frame_count_target:
                    clip_id = _save_recorded_clip(config, label, recorded_frames, frame_size)
                    if clip_id:
                        saved_count += 1
                    recorded_frames = []
                    frame_size = None
                    segment_counter += 1
                    cooldown_frames_remaining = gap_frame_target
                    status_line = f"已儲存 {clip_id or '片段'}"
                    hint_line = "進入 1 秒間隔，之後自動錄下一段。"
            else:
                status_line = "待機中"
                hint_line = f"{_label_specific_hint(label)} 按空白鍵開始連續錄製。"
            _draw_overlay(
                preview,
                label=label,
                label_index=label_index,
                total_labels=len(labels),
                saved_count=saved_count,
                status_line=status_line,
                hint_line=hint_line,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                if is_recording:
                    clip_id = _save_recorded_clip(config, label, recorded_frames, frame_size)
                    if clip_id:
                        saved_count += 1
                    recorded_frames = []
                    frame_size = None
                    is_recording = False
                    cooldown_frames_remaining = 0
                    segment_counter = 0
                    continue

                for seconds_left in range(config.countdown_seconds, 0, -1):
                    ok, countdown_frame = capture.read()
                    if not ok:
                        raise RuntimeError("倒數期間無法讀取 webcam 畫面。")
                    if config.mirror_preview:
                        countdown_frame = cv2.flip(countdown_frame, 1)
                    countdown_preview = countdown_frame.copy()
                    _draw_overlay(
                        countdown_preview,
                        label=label,
                        label_index=label_index,
                        total_labels=len(labels),
                        saved_count=saved_count,
                        status_line=f"準備連續錄製，倒數 {seconds_left}",
                        hint_line=f"{_label_specific_hint(label)} 開始後會自動連續存檔。",
                    )
                    cv2.imshow(window_name, countdown_preview)
                    cv2.waitKey(1000)

                is_recording = True
                cooldown_frames_remaining = 0
                recorded_frames = []
                frame_size = None
                segment_counter = 0
                continue
            if key == ord("n"):
                if is_recording:
                    continue
                label_index = min(len(labels) - 1, label_index + 1)
                continue
            if key == ord("p"):
                if is_recording:
                    continue
                label_index = max(0, label_index - 1)
                continue
    finally:
        if is_recording:
            _save_recorded_clip(config, labels[label_index], recorded_frames, frame_size)
        capture.release()
        cv2.destroyAllWindows()
