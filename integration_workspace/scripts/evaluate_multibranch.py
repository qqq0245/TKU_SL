from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.dataset_reader import SignSequenceDataset, load_split_records
from src.models.feature_slices import split_feature_tensor
from src.models.inference_utils_multibranch import load_multibranch_checkpoint


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate a multibranch checkpoint on a split CSV.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_multibranch_checkpoint(args.checkpoint, device)
    if model is None or checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    split_records = load_split_records(Path(args.split_csv))
    label_to_index = checkpoint["label_to_index"]
    index_to_label = {
        int(index): label for index, label in checkpoint["index_to_label"].items()
    }
    dataset = SignSequenceDataset(Path(args.data_dir), split_records, label_to_index)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total = 0
    correct = 0
    per_class = {
        label: {"correct": 0, "total": 0}
        for label in label_to_index
    }
    prediction_rows = []

    for batch_index, (sequences, labels) in enumerate(loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        streams = split_feature_tensor(
            sequences,
            checkpoint["feature_mode"],
            feature_spec=checkpoint.get("feature_spec"),
        )
        logits = model(
            streams["skeleton_stream"],
            streams["location_stream"],
            streams["motion_stream"],
        )
        predictions = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        for row_offset in range(labels.size(0)):
            record = split_records[batch_index * args.batch_size + row_offset]
            true_index = int(labels[row_offset].item())
            pred_index = int(predictions[row_offset].item())
            true_label = index_to_label[true_index]
            pred_label = index_to_label[pred_index]
            confidence = float(probs[row_offset, pred_index].item())
            is_correct = pred_index == true_index
            total += 1
            correct += int(is_correct)
            per_class[true_label]["total"] += 1
            per_class[true_label]["correct"] += int(is_correct)
            prediction_rows.append(
                {
                    "sample_id": record.get("sample_id", ""),
                    "source_video_path": record.get("metadata", {}).get("source_video_path", ""),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": f"{confidence:.6f}",
                    "correct": str(is_correct),
                }
            )

    accuracy = correct / max(total, 1)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "split_csv": str(Path(args.split_csv).resolve()),
        "samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "feature_mode": checkpoint["feature_mode"],
        "per_class_accuracy": {
            label: (
                stats["correct"] / stats["total"] if stats["total"] else 0.0
            )
            for label, stats in sorted(per_class.items())
        },
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "source_video_path", "true_label", "pred_label", "confidence", "correct"],
            )
            writer.writeheader()
            writer.writerows(prediction_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
