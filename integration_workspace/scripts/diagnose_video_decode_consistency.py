from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Diagnose OpenCV frame-count consistency for a video.")
    parser.add_argument("--video", required=True, help="Video path to inspect.")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional JSON output path. Defaults to <video_dir>/<stem>_decode_consistency.json.",
    )
    parser.add_argument(
        "--init-holistic",
        action="store_true",
        help="Also initialize HolisticExtractor before probing sequential decode counts.",
    )
    return parser


def _sequential_read_count(video_path: Path, api_preference: int | None = None) -> dict[str, object]:
    if api_preference is None:
        capture = cv2.VideoCapture(str(video_path))
        backend_name = "default"
    else:
        capture = cv2.VideoCapture(str(video_path), api_preference)
        backend_name = str(api_preference)
    opened = capture.isOpened()
    property_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else -1
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    readable_frames = 0
    if opened:
        while True:
            success, _frame = capture.read()
            if not success:
                break
            readable_frames += 1
    capture.release()
    return {
        "backend": backend_name,
        "opened": opened,
        "property_frame_count": property_frame_count,
        "fps": fps,
        "readable_frames": readable_frames,
    }


def _probe_random_access(video_path: Path, indices: list[int], api_preference: int | None = None) -> dict[str, object]:
    if api_preference is None:
        capture = cv2.VideoCapture(str(video_path))
        backend_name = "default"
    else:
        capture = cv2.VideoCapture(str(video_path), api_preference)
        backend_name = str(api_preference)
    opened = capture.isOpened()
    checks: list[dict[str, object]] = []
    if opened:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, _frame = capture.read()
            checks.append(
                {
                    "frame_index": int(frame_index),
                    "success": bool(success),
                }
            )
    capture.release()
    return {
        "backend": backend_name,
        "opened": opened,
        "checks": checks,
    }


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    random_indices = [0, 1, 2, 100, 500, 1000, 1023, 1024, 1100, 1200, 1300]
    backend_specs = [
        ("default", None),
        ("CAP_FFMPEG", getattr(cv2, "CAP_FFMPEG", None)),
        ("CAP_MSMF", getattr(cv2, "CAP_MSMF", None)),
    ]

    sequential_results: list[dict[str, object]] = []
    random_access_results: list[dict[str, object]] = []
    phases: list[tuple[str, object | None]] = [("baseline", None)]
    if args.init_holistic:
        if str(Path(__file__).resolve().parents[1]) not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.landmarks.holistic_extractor import HolisticExtractor

        phases.append(("after_holistic_init", HolisticExtractor()))

    for phase_label, holistic_handle in phases:
        for backend_label, backend_value in backend_specs:
            if backend_value is None and backend_label != "default":
                continue
            sequential = _sequential_read_count(video_path, api_preference=backend_value)
            sequential["backend_label"] = backend_label
            sequential["phase"] = phase_label
            sequential_results.append(sequential)
            random_probe = _probe_random_access(video_path, random_indices, api_preference=backend_value)
            random_probe["backend_label"] = backend_label
            random_probe["phase"] = phase_label
            random_access_results.append(random_probe)
        if holistic_handle is not None:
            holistic_handle.close()

    payload = {
        "video_path": str(video_path),
        "opencv_version": cv2.__version__,
        "sequential_results": sequential_results,
        "random_access_results": random_access_results,
    }

    output_json = Path(args.output_json).resolve() if args.output_json else video_path.with_name(f"{video_path.stem}_decode_consistency.json")
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_json}")
    for result in sequential_results:
        print(
            f"{result['phase']} {result['backend_label']}: opened={result['opened']} "
            f"prop={result['property_frame_count']} read={result['readable_frames']} fps={result['fps']:.6f}"
        )


if __name__ == "__main__":
    main()
