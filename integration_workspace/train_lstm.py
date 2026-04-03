from argparse import ArgumentParser

from config import CONFIG
from src.pipeline.training_pipeline import train_lstm_baseline


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train LSTM sign model from default dataset or split CSVs.")
    parser.add_argument("--data-dir", default=str(CONFIG.processed_dir))
    parser.add_argument("--train-split-csv", default=None)
    parser.add_argument("--val-split-csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_lstm_baseline(
        data_dir=args.data_dir,
        train_split_csv=args.train_split_csv,
        val_split_csv=args.val_split_csv,
    )


if __name__ == "__main__":
    main()
