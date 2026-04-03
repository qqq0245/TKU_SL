from argparse import ArgumentParser

from config import CONFIG
from src.pipeline.capture_pipeline import run_capture_session


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Capture sign sequences from webcam.")
    parser.add_argument("--label", required=True, help="Class label to record.")
    parser.add_argument("--num-sequences", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=CONFIG.sequence_length)
    parser.add_argument("--camera-index", type=int, default=CONFIG.camera_index)
    parser.add_argument("--sample-prefix", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_capture_session(
        label=args.label,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        camera_index=args.camera_index,
        sample_prefix=args.sample_prefix,
    )


if __name__ == "__main__":
    main()
