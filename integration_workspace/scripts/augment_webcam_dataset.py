from __future__ import annotations

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import ensure_dir


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build augmented webcam word dataset with flip and photometric variants.")
    parser.add_argument(
        "--source-root",
        default=str(PROJECT_ROOT / "datasets" / "recorded" / "webcam_30_words"),
    )
    parser.add_argument(
        "--labels-csv",
        default=str(PROJECT_ROOT / "metadata" / "webcam_8_vocabulary.csv"),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "datasets" / "training_variants" / "webcam_8_augmented"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["orig", "flip", "bright", "dim"],
    )
    return parser


def load_labels(labels_csv: Path) -> list[str]:
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return [str(row["english_label"]).strip().lower() for row in csv.DictReader(handle)]


def load_source_metadata(source_root: Path) -> dict[str, dict]:
    metadata_path = source_root / "source_metadata.csv"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(Path(row["video_path"]).resolve()): row for row in rows if row.get("video_path")}


def iter_video_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def adjust_brightness(frame, alpha: float, beta: int):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def write_variant(input_video: Path, output_video: Path, variant_name: str) -> None:
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {input_video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to write video: {output_video}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if variant_name == "orig":
                out_frame = frame
            elif variant_name == "flip":
                out_frame = cv2.flip(frame, 1)
            elif variant_name == "bright":
                out_frame = adjust_brightness(frame, alpha=1.10, beta=14)
            elif variant_name == "dim":
                out_frame = adjust_brightness(frame, alpha=0.90, beta=-12)
            else:
                raise ValueError(f"Unsupported variant: {variant_name}")
            writer.write(out_frame)
    finally:
        capture.release()
        writer.release()


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root)
    labels_csv = Path(args.labels_csv)
    output_root = Path(args.output_root)
    labels = load_labels(labels_csv)
    source_metadata = load_source_metadata(source_root)

    variants = [variant.strip().lower() for variant in args.variants if variant.strip()]
    if not variants:
        raise ValueError("At least one variant must be provided.")
    total_written = 0
    output_metadata_rows: list[dict] = []
    for label in labels:
        source_dir = source_root / label
        output_dir = ensure_dir(output_root / label)
        for input_video in iter_video_files(source_dir):
            stem = input_video.stem
            for variant_name in variants:
                output_video = output_dir / f"{stem}_{variant_name}.mp4"
                write_variant(input_video, output_video, variant_name)
                metadata = source_metadata.get(str(input_video.resolve()), {})
                output_metadata_rows.append(
                    {
                        "video_path": str(output_video.resolve()),
                        "source_origin_path": metadata.get("source_origin_path", str(input_video.resolve())),
                        "source_group_key": metadata.get("source_group_key", str(input_video.resolve())),
                    }
                )
                total_written += 1

    metadata_path = output_root / "source_metadata.csv"
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_path", "source_origin_path", "source_group_key"],
        )
        writer.writeheader()
        writer.writerows(output_metadata_rows)

    print(f"Labels: {len(labels)}")
    print(f"Variants per clip: {len(variants)}")
    print(f"Output root: {output_root}")
    print(f"Written videos: {total_written}")
    print(f"Source metadata: {metadata_path}")


if __name__ == "__main__":
    main()
