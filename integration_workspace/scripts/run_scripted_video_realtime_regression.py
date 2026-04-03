from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from run_sentence_interface import RealtimeTestRecorder, RealtimeWordDecoder, get_decoder_settings_for_mode
from src.app.scripted_video_alignment import ScriptedFrameSignal, build_scripted_alignment, parse_script_tokens_from_path
from src.app.sign_sentence_engine import MultibranchSequenceEngine


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run scripted or continuous realtime-style evaluation on prerecorded videos.")
    parser.add_argument("--mode", choices=("scripted", "continuous"), default="scripted")
    parser.add_argument("--engine-mode", choices=("sliding_window", "trigger_based"), default="sliding_window")
    parser.add_argument(
        "--glob",
        default="i_you_mother_father_techer_sudent_want_like*.mp4",
        help="Video glob under the project root for scripted mode.",
    )
    parser.add_argument(
        "--video",
        default="",
        help="Single video path for continuous mode.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "integration_workspace"
            / "artifacts_webcam9_nosign_seq30s5_iso"
            / "models"
            / "multibranch_baseline.pt"
        ),
    )
    parser.add_argument(
        "--gesture-profile",
        default=str(PROJECT_ROOT / "metadata" / "webcam9_gesture_profiles.json"),
    )
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--mirror-input", action="store_true", default=True)
    parser.add_argument(
        "--output-json",
        default=str(PROJECT_ROOT / "reports" / "regression" / "scripted_video_realtime_regression.json"),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "reports" / "realtime_tests"),
        help="Session output root for continuous mode.",
    )
    parser.add_argument(
        "--sentence-manifest",
        default=str(PROJECT_ROOT / "metadata" / "sentence_video_manifest_50.csv"),
    )
    parser.add_argument(
        "--segments-manifest",
        default=str(PROJECT_ROOT / "metadata" / "sentence_word_segments_manifest_50.csv"),
    )
    parser.add_argument(
        "--scripted-glob",
        default="i_you_mother_father_techer_sudent_want_like*.mp4",
        help="Fallback scripted video glob under the project root.",
    )
    return parser


def _canonical_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _blank_like_label(label: str, status: str) -> str:
    if status in {"invalid_frame", "insufficient_signal", "collecting"}:
        return CONFIG.no_sign_label
    if label in {CONFIG.no_sign_label, CONFIG.unknown_label, "collecting"}:
        return CONFIG.no_sign_label
    return label


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_probability_rows(frame_rows: list[dict[str, str]]) -> list[dict[str, float]]:
    probability_rows: list[dict[str, float]] = []
    for row in frame_rows:
        encoded = str(row.get("assisted_probabilities") or row.get("probabilities") or "").strip()
        if not encoded:
            probability_rows.append({})
            continue
        try:
            parsed = json.loads(encoded)
        except json.JSONDecodeError:
            probability_rows.append({})
            continue
        if not isinstance(parsed, dict):
            probability_rows.append({})
            continue
        probability_rows.append({str(label): float(score) for label, score in parsed.items()})
    return probability_rows


def _to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_sentence_reference(
    video_path: Path,
    sentence_manifest: Path,
    segments_manifest: Path,
    frame_count: int,
) -> tuple[list[str], list[str]]:
    sentence_rows = _load_csv_rows(sentence_manifest)
    segments_rows = _load_csv_rows(segments_manifest)
    canonical_video = _canonical_path(video_path)

    sentence_row = next(
        (
            row
            for row in sentence_rows
            if _canonical_path(row.get("output_video", "")) == canonical_video
            or Path(row.get("output_video", "")).name == video_path.name
        ),
        None,
    )
    if sentence_row is None:
        raise FileNotFoundError(f"Video not found in sentence manifest: {video_path}")

    expected_tokens = json.loads(sentence_row["tokens_json"])
    frame_labels = [CONFIG.no_sign_label for _ in range(frame_count)]
    matched_segments = [
        row
        for row in segments_rows
        if _canonical_path(row.get("source_sentence_video", "")) == canonical_video
        or Path(row.get("source_sentence_video", "")).name == video_path.name
    ]
    if not matched_segments:
        raise FileNotFoundError(f"No segment rows found for video: {video_path}")

    for row in sorted(matched_segments, key=lambda item: int(item["word_index"])):
        token = row["token"].strip().lower()
        start = int(row.get("core_frame_start") or row["frame_start"])
        end = int(row.get("core_frame_end_exclusive") or row["frame_end_exclusive"])
        for frame_index in range(max(0, start), min(frame_count, end)):
            frame_labels[frame_index] = token
    return expected_tokens, frame_labels


def _load_scripted_reference_from_signals(
    video_path: Path,
    frame_signals: list[ScriptedFrameSignal],
    *,
    allowed_labels: set[str],
) -> tuple[list[str], list[str]]:
    expected_tokens = parse_script_tokens_from_path(video_path, allowed_labels=allowed_labels)
    if not expected_tokens:
        raise FileNotFoundError(f"No scripted tokens parsed from filename: {video_path}")
    alignment = build_scripted_alignment(video_path, frame_signals, allowed_labels=allowed_labels)
    if alignment is None:
        raise FileNotFoundError(f"Unable to build scripted alignment for: {video_path}")
    last_frame = frame_signals[-1].frame_index if frame_signals else -1
    frame_labels = [CONFIG.no_sign_label for _ in range(max(last_frame + 1, 0))]
    for segment in alignment.get("segments", []):
        token = str(segment["token"]).strip().lower()
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        for frame_index in range(max(0, start), min(len(frame_labels), end + 1)):
            frame_labels[frame_index] = token
    return expected_tokens, frame_labels


def _levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _compute_wer(reference: list[str], hypothesis: list[str]) -> float:
    return _levenshtein_distance(reference, hypothesis) / max(len(reference), 1)


def _compute_frame_metrics(reference: list[str], hypothesis: list[str]) -> dict[str, object]:
    labels = sorted(set(reference) | set(hypothesis))
    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []

    for label in labels:
        tp = sum(1 for ref, hyp in zip(reference, hypothesis) if ref == label and hyp == label)
        fp = sum(1 for ref, hyp in zip(reference, hypothesis) if ref != label and hyp == label)
        fn = sum(1 for ref, hyp in zip(reference, hypothesis) if ref == label and hyp != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall <= 1e-8 else (2.0 * precision * recall) / (precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for ref in reference if ref == label),
        }
        f1_values.append(f1)

    return {
        "frame_macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "no_sign_recall": per_label.get(CONFIG.no_sign_label, {}).get("recall", 0.0),
        "per_label": per_label,
    }


def _plot_probability_timeline(
    output_path: Path,
    reference_labels: list[str],
    predicted_labels: list[str],
    probability_rows: list[dict[str, float]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for continuous evaluation plots.") from exc

    frame_count = min(len(reference_labels), len(predicted_labels), len(probability_rows))
    if frame_count <= 0:
        return
    reference_labels = reference_labels[:frame_count]
    predicted_labels = predicted_labels[:frame_count]
    probability_rows = probability_rows[:frame_count]

    labels = list(
        dict.fromkeys(
            [CONFIG.no_sign_label, *sorted(set(reference_labels) | set(predicted_labels) - {CONFIG.no_sign_label})]
        )
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    sample_step = max(1, (frame_count + 1999) // 2000)
    sampled_indices = list(range(0, frame_count, sample_step))
    if sampled_indices[-1] != frame_count - 1:
        sampled_indices.append(frame_count - 1)

    frames = sampled_indices
    reference_series = [label_to_index[reference_labels[index]] for index in sampled_indices]
    predicted_series = [label_to_index[predicted_labels[index]] for index in sampled_indices]
    gt_probabilities = [
        float(probability_rows[index].get(reference_labels[index], 0.0))
        for index in sampled_indices
    ]
    pred_probabilities = [
        float(probability_rows[index].get(predicted_labels[index], 0.0))
        for index in sampled_indices
    ]
    nosign_probabilities = [
        float(probability_rows[index].get(CONFIG.no_sign_label, 0.0))
        for index in sampled_indices
    ]

    figure, (ax_labels, ax_probs) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    ax_labels.step(frames, reference_series, where="post", label="ground_truth", linewidth=2.0)
    ax_labels.step(frames, predicted_series, where="post", label="predicted", linewidth=1.2, alpha=0.85)
    ax_labels.set_yticks(list(label_to_index.values()))
    ax_labels.set_yticklabels(labels)
    ax_labels.set_title("Ground Truth vs Predicted Labels")
    ax_labels.legend(loc="upper right")
    ax_labels.grid(alpha=0.25)

    ax_probs.plot(frames, gt_probabilities, label="p(ground_truth_label)", linewidth=2.0)
    ax_probs.plot(frames, pred_probabilities, label="p(predicted_label)", linewidth=1.8)
    ax_probs.plot(frames, nosign_probabilities, label="p(no_sign)", linewidth=1.4, alpha=0.8)
    ax_probs.set_ylim(0.0, 1.05)
    ax_probs.set_xlabel("Frame")
    ax_probs.set_ylabel("Probability")
    ax_probs.set_title("Probability Timeline")
    ax_probs.legend(loc="upper right")
    ax_probs.grid(alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_reference_overlap(reference_frame_labels: list[str], start_frame: int, end_frame: int) -> tuple[str, float, dict[str, int]]:
    if end_frame < start_frame or start_frame >= len(reference_frame_labels):
        return CONFIG.no_sign_label, 0.0, {}
    clipped_start = max(0, start_frame)
    clipped_end = min(len(reference_frame_labels) - 1, end_frame)
    if clipped_end < clipped_start:
        return CONFIG.no_sign_label, 0.0, {}
    span = reference_frame_labels[clipped_start : clipped_end + 1]
    counts = Counter(label for label in span if label != CONFIG.no_sign_label)
    if not counts:
        return CONFIG.no_sign_label, 0.0, {}
    label, count = counts.most_common(1)[0]
    return label, count / max(len(span), 1), dict(counts)


def _build_trigger_segment_reports(
    *,
    trigger_segments_csv: Path,
    output_dir: Path,
    reference_frame_labels: list[str],
) -> dict[str, object]:
    if not trigger_segments_csv.exists():
        return {}
    segment_rows = _load_csv_rows(trigger_segments_csv)
    if not segment_rows:
        return {}

    alignment_rows: list[dict[str, object]] = []
    confusion: dict[str, Counter[str]] = {}
    for row in segment_rows:
        start_frame = int(row.get("start_frame") or 0)
        end_frame = int(row.get("end_frame") or start_frame)
        reference_label, overlap_ratio, overlap_counts = _summarize_reference_overlap(
            reference_frame_labels,
            start_frame,
            end_frame,
        )
        predicted_label = str(row.get("emitted_label") or row.get("raw_label") or CONFIG.no_sign_label).strip().lower()
        segment_confusion = f"{reference_label}->{predicted_label}"
        confusion.setdefault(reference_label, Counter())[predicted_label] += 1
        alignment_rows.append(
            {
                "segment_id": int(row.get("segment_id") or 0),
                "raw_start_frame": int(row.get("raw_start_frame") or start_frame),
                "raw_end_frame": int(row.get("raw_end_frame") or end_frame),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "decision_frame": int(row.get("decision_frame") or end_frame),
                "raw_length": int(row.get("raw_length") or 0),
                "trimmed_length": int(row.get("trimmed_length") or 0),
                "pre_context_frames": int(row.get("pre_context_frames") or 0),
                "tail_trimmed": int(row.get("tail_trimmed") or 0),
                "tail_trimmed_no_hand": int(row.get("tail_trimmed_no_hand") or 0),
                "end_reason": str(row.get("end_reason") or ""),
                "trigger_end_reason": str(row.get("trigger_end_reason") or row.get("end_reason") or ""),
                "reference_label": reference_label,
                "reference_overlap_ratio": f"{overlap_ratio:.6f}",
                "reference_overlap_counts": json.dumps(overlap_counts, ensure_ascii=False),
                "raw_label": str(row.get("raw_label") or ""),
                "raw_confidence": row.get("raw_confidence") or "0.000000",
                "emitted_label": str(row.get("emitted_label") or ""),
                "decision_status": str(row.get("decision_status") or ""),
                "segment_confusion": segment_confusion,
                "top_margin": row.get("top_margin") or "0.000000",
                "sampled_indices": row.get("sampled_indices") or "[]",
                "sampled_frame_indices": row.get("sampled_frame_indices") or "[]",
                "motion_energy_curve": row.get("motion_energy_curve") or "[]",
                "top_candidates": row.get("top_candidates") or "[]",
                "top_logits": row.get("top_logits") or "[]",
                "sampled_path_top_candidates": row.get("sampled_path_top_candidates") or "[]",
                "sampled_path_top_logits": row.get("sampled_path_top_logits") or "[]",
                "sampled_path_raw_label": row.get("sampled_path_raw_label") or "",
                "sampled_path_raw_confidence": row.get("sampled_path_raw_confidence") or "0.000000",
                "feature_path_l2_mean": row.get("feature_path_l2_mean") or "0.000000",
                "feature_path_l2_max": row.get("feature_path_l2_max") or "0.000000",
                "feature_path_cosine_mean": row.get("feature_path_cosine_mean") or "0.000000",
                "segment_has_left_hand": row.get("segment_has_left_hand") or "0",
                "segment_has_right_hand": row.get("segment_has_right_hand") or "0",
                "segment_has_both_hands": row.get("segment_has_both_hands") or "0",
                "segment_pose_ratio": row.get("segment_pose_ratio") or "0.000000",
                "you_like_frontload_attempted": row.get("you_like_frontload_attempted") or "0",
                "you_like_frontload_applied": row.get("you_like_frontload_applied") or "0",
                "you_like_frontload_power": row.get("you_like_frontload_power") or "0.000000",
                "you_like_frontload_top_margin": row.get("you_like_frontload_top_margin") or "0.000000",
                "you_like_frontload_raw_confidence": row.get("you_like_frontload_raw_confidence") or "0.000000",
                "you_like_frontload_top_candidates": row.get("you_like_frontload_top_candidates") or "[]",
                "you_like_frontload_sampled_indices": row.get("you_like_frontload_sampled_indices") or "[]",
                "you_like_frontload_sampled_frame_indices": row.get("you_like_frontload_sampled_frame_indices") or "[]",
                "you_like_pairwise_attempted": row.get("you_like_pairwise_attempted") or "0",
                "you_like_pairwise_applied": row.get("you_like_pairwise_applied") or "0",
                "you_like_pairwise_delta": row.get("you_like_pairwise_delta") or "0.000000",
                "you_like_pairwise_top_margin": row.get("you_like_pairwise_top_margin") or "0.000000",
                "you_like_pairwise_raw_confidence": row.get("you_like_pairwise_raw_confidence") or "0.000000",
                "you_like_pairwise_top_candidates": row.get("you_like_pairwise_top_candidates") or "[]",
                "like_i_pairwise_attempted": row.get("like_i_pairwise_attempted") or "0",
                "like_i_pairwise_applied": row.get("like_i_pairwise_applied") or "0",
                "like_i_pairwise_delta": row.get("like_i_pairwise_delta") or "0.000000",
                "like_i_pairwise_top_margin": row.get("like_i_pairwise_top_margin") or "0.000000",
                "like_i_pairwise_raw_confidence": row.get("like_i_pairwise_raw_confidence") or "0.000000",
                "like_i_pairwise_top_candidates": row.get("like_i_pairwise_top_candidates") or "[]",
                "want_anchor_fallback_attempted": row.get("want_anchor_fallback_attempted") or "0",
                "want_anchor_fallback_applied": row.get("want_anchor_fallback_applied") or "0",
                "want_anchor_fallback_raw_label": row.get("want_anchor_fallback_raw_label") or "",
                "want_anchor_fallback_raw_confidence": row.get("want_anchor_fallback_raw_confidence") or "0.000000",
                "want_anchor_fallback_top_margin": row.get("want_anchor_fallback_top_margin") or "0.000000",
                "want_anchor_fallback_top_candidates": row.get("want_anchor_fallback_top_candidates") or "[]",
                "want_anchor_fallback_candidate_anchor": row.get("want_anchor_fallback_candidate_anchor") or "",
                "father_trigger_rescue_attempted": row.get("father_trigger_rescue_attempted") or "0",
                "father_trigger_rescue_applied": row.get("father_trigger_rescue_applied") or "0",
                "father_trigger_rescue_hit_ratio": row.get("father_trigger_rescue_hit_ratio") or "0.000000",
                "father_trigger_rescue_valid_count": row.get("father_trigger_rescue_valid_count") or "0",
                "father_trigger_rescue_pairwise_delta": row.get("father_trigger_rescue_pairwise_delta") or "0.000000",
                "father_trigger_rescue_raw_label": row.get("father_trigger_rescue_raw_label") or "",
                "father_trigger_rescue_raw_confidence": row.get("father_trigger_rescue_raw_confidence") or "0.000000",
                "father_trigger_rescue_top_margin": row.get("father_trigger_rescue_top_margin") or "0.000000",
                "father_trigger_rescue_top_candidates": row.get("father_trigger_rescue_top_candidates") or "[]",
            }
        )

    alignment_csv = output_dir / "trigger_segment_alignment.csv"
    confusion_json = output_dir / "trigger_segment_confusion.json"
    _write_csv(
        alignment_csv,
        fieldnames=[
            "segment_id",
            "raw_start_frame",
            "raw_end_frame",
            "start_frame",
            "end_frame",
            "decision_frame",
            "raw_length",
            "trimmed_length",
            "pre_context_frames",
            "tail_trimmed",
            "tail_trimmed_no_hand",
            "end_reason",
            "trigger_end_reason",
            "reference_label",
            "reference_overlap_ratio",
            "reference_overlap_counts",
            "raw_label",
            "raw_confidence",
            "emitted_label",
            "decision_status",
            "segment_confusion",
            "top_margin",
            "sampled_indices",
            "sampled_frame_indices",
            "motion_energy_curve",
            "top_candidates",
            "top_logits",
            "sampled_path_top_candidates",
            "sampled_path_top_logits",
            "sampled_path_raw_label",
            "sampled_path_raw_confidence",
            "feature_path_l2_mean",
            "feature_path_l2_max",
            "feature_path_cosine_mean",
            "segment_has_left_hand",
            "segment_has_right_hand",
            "segment_has_both_hands",
            "segment_pose_ratio",
            "you_like_frontload_attempted",
            "you_like_frontload_applied",
            "you_like_frontload_power",
            "you_like_frontload_top_margin",
            "you_like_frontload_raw_confidence",
            "you_like_frontload_top_candidates",
            "you_like_frontload_sampled_indices",
            "you_like_frontload_sampled_frame_indices",
            "you_like_pairwise_attempted",
            "you_like_pairwise_applied",
            "you_like_pairwise_delta",
            "you_like_pairwise_top_margin",
            "you_like_pairwise_raw_confidence",
            "you_like_pairwise_top_candidates",
            "like_i_pairwise_attempted",
            "like_i_pairwise_applied",
            "like_i_pairwise_delta",
            "like_i_pairwise_top_margin",
            "like_i_pairwise_raw_confidence",
            "like_i_pairwise_top_candidates",
            "want_anchor_fallback_attempted",
            "want_anchor_fallback_applied",
            "want_anchor_fallback_raw_label",
            "want_anchor_fallback_raw_confidence",
            "want_anchor_fallback_top_margin",
            "want_anchor_fallback_top_candidates",
            "want_anchor_fallback_candidate_anchor",
            "father_trigger_rescue_attempted",
            "father_trigger_rescue_applied",
            "father_trigger_rescue_hit_ratio",
            "father_trigger_rescue_valid_count",
            "father_trigger_rescue_pairwise_delta",
            "father_trigger_rescue_raw_label",
            "father_trigger_rescue_raw_confidence",
            "father_trigger_rescue_top_margin",
            "father_trigger_rescue_top_candidates",
        ],
        rows=alignment_rows,
    )
    confusion_payload = {
        "labels": sorted(confusion.keys()),
        "confusion": {
            ref_label: dict(sorted(pred_counts.items()))
            for ref_label, pred_counts in sorted(confusion.items())
        },
    }
    confusion_json.write_text(json.dumps(confusion_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "alignment_csv": str(alignment_csv),
        "confusion_json": str(confusion_json),
        "segment_count": len(alignment_rows),
    }


def evaluate_video(video_path: Path, engine: MultibranchSequenceEngine, mirror_input: bool) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frame_signals: list[ScriptedFrameSignal] = []
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if mirror_input:
                frame = cv2.flip(frame, 1)
            prediction = engine.process_frame(frame)
            frame_signals.append(
                ScriptedFrameSignal(
                    frame_index=prediction.frame_index,
                    signal_score=prediction.signal_score,
                    motion_energy=prediction.motion_energy,
                    status=prediction.decision.status,
                    left_hand_present=prediction.left_hand_present,
                    right_hand_present=prediction.right_hand_present,
                    pose_present=prediction.pose_present,
                )
            )
            frame_count += 1
    finally:
        capture.release()

    alignment = build_scripted_alignment(
        video_path,
        frame_signals,
        allowed_labels=set(engine.index_to_label.values()),
    )
    if alignment is None:
        raise RuntimeError(f"No scripted tokens parsed from filename: {video_path.name}")
    return {
        "video_name": video_path.name,
        "frame_count": frame_count,
        **alignment,
    }


def run_scripted_mode(args) -> None:
    matched_video_paths = sorted(PROJECT_ROOT.glob(args.glob))
    if not matched_video_paths:
        raise FileNotFoundError(f"No videos matched: {args.glob}")
    video_paths = []
    for video_path in matched_video_paths:
        if not video_path.exists():
            print(f"[WARNING] Skipping missing video: {video_path}")
            continue
        video_paths.append(video_path)
    if not video_paths:
        raise FileNotFoundError(f"No existing videos matched: {args.glob}")

    checkpoint_path = Path(args.checkpoint)
    gesture_profile_path = Path(args.gesture_profile)
    results: list[dict[str, object]] = []
    for video_path in video_paths:
        engine = MultibranchSequenceEngine(
            checkpoint_path,
            int(args.sequence_length),
            float(args.confidence_threshold),
            gesture_profile_path=gesture_profile_path,
            mode=str(args.engine_mode),
        )
        try:
            results.append(evaluate_video(video_path, engine, bool(args.mirror_input)))
        finally:
            engine.close()

    summary = {
        "mode": "scripted",
        "checkpoint": str(checkpoint_path),
        "gesture_profile": str(gesture_profile_path),
        "video_count": len(results),
        "pass_count": sum(1 for item in results if item.get("pass")),
        "results": results,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for item in results:
        expected = " / ".join(item.get("expected_tokens", []))
        actual = " / ".join(item.get("tokens", []))
        status = "PASS" if item.get("pass") else "FAIL"
        print(f"{status} | {item['video_name']}")
        print(f"  expected: {expected}")
        print(f"  aligned : {actual}")
        print(
            f"  segments: detected={item.get('detected_segment_count')} "
            f"aligned={item.get('aligned_segment_count')}"
        )
    print(f"Summary: {summary['pass_count']}/{summary['video_count']} passed")
    print(f"Saved: {output_json}")


def run_continuous_mode(args) -> None:
    if not args.video:
        raise ValueError("--video is required in continuous mode.")

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[WARNING] Skipping missing video: {video_path}")
        return

    checkpoint_path = Path(args.checkpoint)
    gesture_profile_path = Path(args.gesture_profile)
    engine = MultibranchSequenceEngine(
        checkpoint_path,
        int(args.sequence_length),
        float(args.confidence_threshold),
        gesture_profile_path=gesture_profile_path,
        mode=str(args.engine_mode),
    )
    decoder = None
    decoder_params: dict[str, object]
    if str(args.engine_mode) == "trigger_based":
        decoder_params = {
            "kind": "trigger_based",
            "emit_confidence_threshold": float(engine.trigger_emit_confidence_threshold),
            "sequence_length": int(args.sequence_length),
            "pose_local_anchor": str(engine.pose_local_anchor),
            "enable_mother_nose_z_calibration": bool(engine.enable_mother_nose_z_calibration),
            "mother_nose_z_offset": float(engine.mother_nose_z_offset),
            "mother_nose_z_prototypes_ready": bool(engine.mother_you_left_face_prototypes),
            "enable_you_i_leftbody_calibration": bool(engine.enable_you_i_leftbody_calibration),
            "you_i_leftbody_y_offset": float(engine.you_i_leftbody_y_offset),
            "you_i_leftbody_prototypes_ready": bool(engine.you_i_left_body_prototypes),
            "enable_you_like_pairwise_calibration": bool(engine.enable_you_like_pairwise_calibration),
            "you_like_pairwise_delta": float(engine.you_like_pairwise_delta),
            "enable_like_i_pairwise_calibration": bool(engine.enable_like_i_pairwise_calibration),
            "like_i_pairwise_delta": float(engine.like_i_pairwise_delta),
            "enable_you_like_frontloaded_rescore": bool(engine.enable_you_like_frontloaded_rescore),
            "you_like_frontload_power": float(engine.you_like_frontload_power),
            "enable_want_anchor_fallback": bool(engine.enable_want_anchor_fallback),
            "want_anchor_fallback_max_nosign_confidence": float(engine.want_anchor_fallback_max_nosign_confidence),
            "enable_father_trigger_rescue": bool(engine.enable_father_trigger_rescue),
            "father_trigger_rescue_pairwise_delta": float(engine.father_trigger_rescue_pairwise_delta),
            "father_trigger_rescue_min_ratio": float(engine.father_trigger_rescue_min_ratio),
            "father_trigger_rescue_max_nosign_confidence": float(engine.father_trigger_rescue_max_nosign_confidence),
        }
    else:
        decoder_settings = get_decoder_settings_for_mode(
            "webcam 9 類即時模式",
            confidence_threshold=float(args.confidence_threshold),
        )
        decoder = RealtimeWordDecoder(**decoder_settings)
        decoder_params = decoder.describe()
    recorder = RealtimeTestRecorder(
        output_root=Path(args.output_root),
        mode_name="continuous_evaluation",
        checkpoint_path=str(checkpoint_path),
        confidence_threshold=float(args.confidence_threshold),
        sequence_length=int(args.sequence_length),
        mirror_input=bool(args.mirror_input),
        camera_index=-1,
        gesture_profile_path=str(gesture_profile_path),
        decoder_params=decoder_params,
        source_video_path=str(video_path),
    )

    emitted_tokens: list[str] = []
    predicted_frame_labels: list[str] = []
    frame_signals: list[ScriptedFrameSignal] = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        recorder.close()
        engine.close()
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.mirror_input:
                frame = cv2.flip(frame, 1)
            prediction = engine.process_frame(frame)
            frame_signals.append(
                ScriptedFrameSignal(
                    frame_index=prediction.frame_index,
                    signal_score=prediction.signal_score,
                    motion_energy=prediction.motion_energy,
                    status=prediction.decision.status,
                    left_hand_present=prediction.left_hand_present,
                    right_hand_present=prediction.right_hand_present,
                    pose_present=prediction.pose_present,
                )
            )
            if str(args.engine_mode) == "trigger_based":
                emitted_label = prediction.emitted_label
                if emitted_label:
                    emitted_tokens.append(emitted_label)
                    stream_label = emitted_label
                    stream_confidence = float(prediction.decision.final_confidence)
                    decoder_state = prediction.decision.status
                else:
                    stream_label = "收集中" if prediction.decision.status == "trigger_collecting" else "待機中"
                    stream_confidence = 0.0
                    decoder_state = prediction.decision.status
            else:
                assert decoder is not None
                stream_label, stream_confidence, emitted_label, decoder_state = decoder.update(prediction)
                if emitted_label:
                    emitted_tokens.append(emitted_label)
            frame_label = _blank_like_label(prediction.decision.final_label, prediction.decision.status)
            predicted_frame_labels.append(frame_label)
            recorder.write_frame(
                prediction.display_frame,
                frame_index=prediction.frame_index,
                raw_label=prediction.decision.raw_label,
                raw_confidence=prediction.decision.raw_confidence,
                final_label=prediction.decision.final_label,
                final_confidence=prediction.decision.final_confidence,
                status=prediction.decision.status,
                valid_ratio=prediction.decision.valid_ratio,
                top_candidates=prediction.top_candidates,
                stream_label=stream_label,
                stream_confidence=stream_confidence,
                decoder_state=decoder_state,
                emitted_label=emitted_label or "",
                assisted_top_candidates=prediction.assisted_top_candidates,
                assist_notes=prediction.assist_notes,
                probabilities=prediction.probabilities,
                assisted_probabilities=prediction.assisted_probabilities,
                disambiguation_notes=prediction.disambiguation_notes,
                disambiguation_applied=prediction.disambiguation_applied,
                motion_energy=prediction.motion_energy,
                top_margin=prediction.top_margin,
                signal_score=prediction.signal_score,
                left_hand_present=prediction.left_hand_present,
                right_hand_present=prediction.right_hand_present,
                pose_present=prediction.pose_present,
                trigger_segment_debug=prediction.trigger_segment_debug,
            )
    finally:
        capture.release()
        saved = recorder.close()
        engine.close()

    try:
        reference_tokens, reference_frame_labels = _load_sentence_reference(
            video_path,
            Path(args.sentence_manifest),
            Path(args.segments_manifest),
            frame_count=len(predicted_frame_labels),
        )
    except FileNotFoundError:
        reference_tokens, reference_frame_labels = _load_scripted_reference_from_signals(
            video_path,
            frame_signals,
            allowed_labels=set(engine.index_to_label.values()),
        )
    wer = _compute_wer(reference_tokens, emitted_tokens)
    frame_metrics = _compute_frame_metrics(reference_frame_labels, predicted_frame_labels)
    plot_path = Path(saved["session_dir"]) / "label_probability_timeline.png"
    plot_error = ""
    try:
        frame_rows = _load_csv_rows(Path(saved["frame_csv_path"]))
        probability_rows = _extract_probability_rows(frame_rows)
        _plot_probability_timeline(plot_path, reference_frame_labels, predicted_frame_labels, probability_rows)
    except (MemoryError, OSError, RuntimeError, ValueError) as exc:
        plot_error = f"{type(exc).__name__}: {exc}"

    summary_path = Path(saved["summary_json_path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trigger_segment_report = _build_trigger_segment_reports(
        trigger_segments_csv=Path(saved["session_dir"]) / "trigger_segments.csv",
        output_dir=Path(saved["session_dir"]),
        reference_frame_labels=reference_frame_labels,
    )
    summary["continuous_evaluation"] = {
        "reference_tokens": reference_tokens,
        "predicted_tokens": emitted_tokens,
        "word_error_rate": wer,
        "frame_macro_f1": frame_metrics["frame_macro_f1"],
        "no_sign_recall": frame_metrics["no_sign_recall"],
        "per_label_frame_metrics": frame_metrics["per_label"],
        "label_probability_plot": str(plot_path) if plot_path.exists() else "",
        "label_probability_plot_error": plot_error,
        "trigger_segment_report": trigger_segment_report,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Video: {video_path.name}")
    print(f"Reference tokens: {' / '.join(reference_tokens)}")
    print(f"Predicted tokens: {' / '.join(emitted_tokens) if emitted_tokens else '-'}")
    print(f"WER: {wer:.4f}")
    print(f"Frame macro F1: {frame_metrics['frame_macro_f1']:.4f}")
    print(f"no_sign recall: {frame_metrics['no_sign_recall']:.4f}")
    print(f"Session: {saved['session_dir']}")
    print(f"Plot: {plot_path}")


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "continuous":
        run_continuous_mode(args)
        return
    run_scripted_mode(args)


if __name__ == "__main__":
    main()
