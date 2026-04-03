from argparse import ArgumentParser

from config import CONFIG
from src.pipeline.capture_pipeline import run_realtime_inference


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run realtime inference from webcam.")
    parser.add_argument("--checkpoint", default=str(CONFIG.models_dir / "lstm_baseline.pt"))
    parser.add_argument("--sequence-length", type=int, default=CONFIG.sequence_length)
    parser.add_argument("--camera-index", type=int, default=CONFIG.camera_index)
    parser.add_argument("--confidence-threshold", type=float, default=CONFIG.confidence_threshold)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_realtime_inference(
        checkpoint_path=args.checkpoint,
        sequence_length=args.sequence_length,
        camera_index=args.camera_index,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
