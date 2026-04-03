from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.landmarks.feature_builder import build_frame_feature
from src.landmarks.holistic_extractor import HolisticExtractor
from src.models.feature_slices import split_feature_tensor
from src.models.inference_utils_multibranch import load_multibranch_checkpoint


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Diagnose continuous inference parity / branch skew on aligned trigger spans.")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "artifacts_webcam9_relative_coord_v1" / "models" / "multibranch_baseline.pt"),
        help="Multibranch checkpoint path.",
    )
    parser.add_argument(
        "--session-dir",
        default="",
        help="Realtime session directory that contains trigger_segment_alignment.csv and session_summary.json.",
    )
    parser.add_argument(
        "--video",
        default="",
        help="Optional video path. Defaults to session_summary.json source_video_path when --session-dir is set.",
    )
    parser.add_argument(
        "--segments-csv",
        default="",
        help="Optional alignment CSV. Defaults to <session-dir>/trigger_segment_alignment.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to <session-dir>/parity_diagnostics.",
    )
    return parser


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_video_frames(video_path: Path) -> tuple[list[np.ndarray], int]:
    capture = cv2.VideoCapture(str(video_path), getattr(cv2, "CAP_FFMPEG", 0))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    property_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Video has zero readable frames: {video_path}")
    return frames, property_frame_count


def _build_feature_cache(
    *,
    frames: list[np.ndarray],
    mirror_input: bool,
    swap_hands: bool,
    feature_mode: str,
    feature_spec: dict | None,
) -> list[np.ndarray]:
    extractor = HolisticExtractor()
    cached: list[np.ndarray] = []
    prev_feature = None
    try:
        for frame in frames:
            current = cv2.flip(frame, 1) if mirror_input else frame
            landmarks, _ = extractor.extract(current)
            left_hand = landmarks.right_hand if swap_hands else landmarks.left_hand
            right_hand = landmarks.left_hand if swap_hands else landmarks.right_hand
            left_hand_mask = landmarks.right_hand_mask if swap_hands else landmarks.left_hand_mask
            right_hand_mask = landmarks.left_hand_mask if swap_hands else landmarks.right_hand_mask
            built = build_frame_feature(
                left_hand=left_hand,
                right_hand=right_hand,
                pose=landmarks.pose,
                mouth_center=landmarks.mouth_center,
                chin=landmarks.chin,
                left_hand_mask=left_hand_mask,
                right_hand_mask=right_hand_mask,
                pose_mask=landmarks.pose_mask,
                mouth_mask=landmarks.mouth_mask,
                chin_mask=landmarks.chin_mask,
                prev_feature=prev_feature,
                feature_mode=feature_mode,
                feature_spec=feature_spec,
            )
            prev_feature = built
            cached.append(built["feature_vector"].astype(np.float32))
    finally:
        extractor.close()
    return cached


def _sample_sequence(features: list[np.ndarray], start_frame: int, end_frame: int, sequence_length: int = 30) -> np.ndarray:
    span = features[start_frame : end_frame + 1]
    if not span:
        raise ValueError(f"Invalid empty span start={start_frame} end={end_frame}")
    stacked = np.stack(span, axis=0).astype(np.float32)
    if len(stacked) == sequence_length:
        return stacked
    if len(stacked) == 1:
        return np.repeat(stacked, sequence_length, axis=0)
    sampled_indices = np.linspace(0, len(stacked) - 1, sequence_length).round().astype(int)
    return stacked[sampled_indices]


@torch.no_grad()
def _predict_variant(
    *,
    model,
    index_to_label: dict[int, str],
    feature_mode: str,
    feature_spec: dict | None,
    sequence: np.ndarray,
    variant: str,
) -> tuple[str, float, list[tuple[str, float]]]:
    ablated = sequence.copy()
    parts_np = split_feature_tensor(ablated, feature_mode, feature_spec=feature_spec)
    if variant in {"zero_location", "zero_location_motion"} and parts_np["location_stream"] is not None:
        parts_np["location_stream"][...] = 0.0
    if variant in {"zero_motion", "zero_location_motion"} and parts_np["motion_stream"] is not None:
        parts_np["motion_stream"][...] = 0.0

    tensor = torch.from_numpy(ablated).unsqueeze(0)
    parts = split_feature_tensor(tensor, feature_mode, feature_spec=feature_spec)
    logits = model(parts["skeleton_stream"], parts["location_stream"], parts["motion_stream"])
    probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    top_indices = np.argsort(-probs)[:3]
    ranked = [(index_to_label[int(index)], float(probs[int(index)])) for index in top_indices]
    label, confidence = ranked[0]
    return label, confidence, ranked


def _default_output_dir(session_dir: Path | None) -> Path:
    if session_dir is not None:
        return session_dir / "parity_diagnostics"
    return PROJECT_ROOT / "reports" / "continuous_inference_repair" / "parity_diagnostics"


def main() -> None:
    args = build_parser().parse_args()
    session_dir = Path(args.session_dir).resolve() if args.session_dir else None
    checkpoint_path = Path(args.checkpoint).resolve()

    session_summary: dict[str, object] = {}
    if session_dir is not None:
        summary_path = session_dir / "session_summary.json"
        if summary_path.exists():
            session_summary = _load_json(summary_path)

    video_path = Path(args.video).resolve() if args.video else None
    if video_path is None and session_summary.get("source_video_path"):
        video_path = Path(str(session_summary["source_video_path"])).resolve()
    if video_path is None:
        raise RuntimeError("video path is required when session_summary.json is unavailable")

    segments_csv = Path(args.segments_csv).resolve() if args.segments_csv else None
    if segments_csv is None and session_dir is not None:
        segments_csv = session_dir / "trigger_segment_alignment.csv"
    if segments_csv is None or not segments_csv.exists():
        raise RuntimeError("segments CSV is required and must exist")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(session_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, checkpoint = load_multibranch_checkpoint(checkpoint_path, device)
    if model is None or checkpoint is None:
        raise RuntimeError(f"Unable to load checkpoint: {checkpoint_path}")
    model.eval()

    frames, property_frame_count = _load_video_frames(video_path)
    frame_count = len(frames)
    feature_mode = str(checkpoint["feature_mode"])
    feature_spec = checkpoint.get("feature_spec")
    index_to_label = checkpoint["index_to_label"]

    raw_rows = _read_csv_rows(segments_csv)
    valid_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, object]] = []
    for row in raw_rows:
        reference_label = row.get("reference_label", "").strip().lower()
        if not reference_label:
            continue
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        if start_frame < 0 or end_frame < start_frame:
            skipped_rows.append(
                {
                    "segment_id": row.get("segment_id", ""),
                    "reason": "invalid_span",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
            continue
        if start_frame >= frame_count:
            skipped_rows.append(
                {
                    "segment_id": row.get("segment_id", ""),
                    "reason": "start_oob",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_count": frame_count,
                }
            )
            continue
        row["effective_end_frame"] = str(min(end_frame, frame_count - 1))
        if int(row["effective_end_frame"]) != end_frame:
            skipped_rows.append(
                {
                    "segment_id": row.get("segment_id", ""),
                    "reason": "end_clamped",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "effective_end_frame": int(row["effective_end_frame"]),
                    "frame_count": frame_count,
                }
            )
        valid_rows.append(row)

    variants = ["full", "zero_location", "zero_motion", "zero_location_motion"]
    mirror_modes = [False, True]
    hand_layouts = [False, True]
    feature_cache = {
        (mirror_input, swap_hands): _build_feature_cache(
            frames=frames,
            mirror_input=mirror_input,
            swap_hands=swap_hands,
            feature_mode=feature_mode,
            feature_spec=feature_spec,
        )
        for mirror_input in mirror_modes
        for swap_hands in hand_layouts
    }

    output_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for mirror_input in mirror_modes:
        for swap_hands in hand_layouts:
            for variant in variants:
                correct_count = 0
                attractor_counter: Counter[str] = Counter()
                per_reference: dict[str, Counter[str]] = defaultdict(Counter)
                total_count = 0
                for row in valid_rows:
                    start_frame = int(row["start_frame"])
                    end_frame = int(row["effective_end_frame"])
                    reference_label = row["reference_label"].strip().lower()
                    sequence = _sample_sequence(feature_cache[(mirror_input, swap_hands)], start_frame, end_frame)
                    predicted_label, predicted_confidence, ranked = _predict_variant(
                        model=model,
                        index_to_label=index_to_label,
                        feature_mode=feature_mode,
                        feature_spec=feature_spec,
                        sequence=sequence,
                        variant=variant,
                    )
                    correct = predicted_label == reference_label
                    correct_count += int(correct)
                    total_count += 1
                    attractor_counter[predicted_label] += 1
                    per_reference[reference_label][predicted_label] += 1
                    output_rows.append(
                        {
                            "segment_id": row.get("segment_id", ""),
                            "reference_label": reference_label,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "baseline_segment_confusion": row.get("segment_confusion", ""),
                            "mirror_input": int(mirror_input),
                            "swap_hands": int(swap_hands),
                            "variant": variant,
                            "predicted_label": predicted_label,
                            "predicted_confidence": f"{predicted_confidence:.6f}",
                            "correct": int(correct),
                            "top3_json": json.dumps(
                                [
                                    {"label": label, "confidence": round(confidence, 6)}
                                    for label, confidence in ranked
                                ],
                                ensure_ascii=False,
                            ),
                        }
                    )
                accuracy = (correct_count / total_count) if total_count else 0.0
                summary_rows.append(
                    {
                        "mirror_input": int(mirror_input),
                        "swap_hands": int(swap_hands),
                        "variant": variant,
                        "segments_evaluated": total_count,
                        "correct_count": correct_count,
                        "accuracy": round(accuracy, 6),
                        "attractor_counts": dict(attractor_counter),
                        "per_reference_confusion": {
                            reference_label: dict(counter)
                            for reference_label, counter in sorted(per_reference.items())
                        },
                    }
                )

    csv_path = output_dir / "segment_variant_predictions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "segment_id",
                "reference_label",
                "start_frame",
                "end_frame",
                "baseline_segment_confusion",
                "mirror_input",
                "swap_hands",
                "variant",
                "predicted_label",
                "predicted_confidence",
                "correct",
                "top3_json",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    summary_path = output_dir / "parity_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "video_path": str(video_path),
                "checkpoint_path": str(checkpoint_path),
                "segments_csv": str(segments_csv),
                "property_frame_count": property_frame_count,
                "frame_count": frame_count,
                "segments_used": len(valid_rows),
                "segments_skipped": skipped_rows,
                "variant_summaries": summary_rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {summary_path}")
    for row in summary_rows:
        print(
            f"mirror={row['mirror_input']} swap_hands={row['swap_hands']} variant={row['variant']} "
            f"accuracy={row['accuracy']:.4f} attractors={json.dumps(row['attractor_counts'], ensure_ascii=False)}"
        )


if __name__ == "__main__":
    main()
