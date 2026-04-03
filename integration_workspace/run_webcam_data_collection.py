from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from src.app.webcam_data_recorder import RecordingConfig, load_default_labels, run_webcam_data_recorder


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Record webcam sign-word videos for realtime fine-tuning.")
    parser.add_argument(
        "--output-root",
        default=r"c:\Users\qqq02\Desktop\99_docs_analysis\datasets\recorded\webcam_30_words",
        help="Root folder where per-label webcam clips will be saved.",
    )
    parser.add_argument(
        "--manifest-csv",
        default=r"c:\Users\qqq02\Desktop\99_docs_analysis\metadata\webcam_30_words_manifest.csv",
        help="CSV manifest appended after each recorded clip.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--countdown-seconds", type=int, default=3)
    parser.add_argument("--clip-seconds", type=float, default=2.4)
    parser.add_argument("--segment-gap-seconds", type=float, default=1.0)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional subset of labels to record. Defaults to 30 words plus no_sign and transition.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = [label.strip().lower() for label in args.labels] if args.labels else load_default_labels()
    config = RecordingConfig(
        output_root=Path(args.output_root),
        manifest_csv=Path(args.manifest_csv),
        camera_index=args.camera_index,
        fps=args.fps,
        width=args.width,
        height=args.height,
        countdown_seconds=args.countdown_seconds,
        clip_seconds=args.clip_seconds,
        segment_gap_seconds=args.segment_gap_seconds,
    )
    run_webcam_data_recorder(config, labels=labels)


if __name__ == "__main__":
    main()
