from __future__ import annotations

import csv
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.app.scripted_video_alignment import ScriptedFrameSignal, build_scripted_alignment, parse_script_tokens_from_path


HARDCASE_FINGER_LABELS = ("i", "like", "you")
HARDCASE_POSITION_LABELS = ("father", "mother")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run offline + continuous post-train evaluation and print a Markdown summary report."
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(ROOT / "artifacts_webcam9_relative_coord_v1"),
        help="Artifact directory that contains models/.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint path. Defaults to <artifact-dir>/models/multibranch_baseline.pt.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Optional processed_sequences directory. Auto-derived from artifact dir when omitted.",
    )
    parser.add_argument(
        "--test-split-csv",
        default="",
        help="Optional test split CSV path. Auto-derived from dataset pipeline when omitted.",
    )
    parser.add_argument(
        "--test-eval-json",
        default="",
        help="Optional existing offline evaluation summary JSON path.",
    )
    parser.add_argument(
        "--test-predictions-csv",
        default="",
        help="Optional existing offline prediction CSV path.",
    )
    parser.add_argument(
        "--force-test-inference",
        action="store_true",
        help="Force rerunning offline test inference even if cached outputs already exist.",
    )
    parser.add_argument(
        "--continuous-video",
        default="",
        help="Optional video path for continuous evaluation. Auto-selects a scripted regression video when omitted.",
    )
    parser.add_argument(
        "--scripted-glob",
        default="i_you_mother_father_techer_sudent_want_like*.mp4",
        help="Fallback scripted regression video glob under the project root.",
    )
    parser.add_argument(
        "--continuous-session-dir",
        default="",
        help="Reuse an existing realtime session directory instead of running a fresh continuous evaluation.",
    )
    parser.add_argument(
        "--skip-continuous-run",
        action="store_true",
        help="Skip launching a new continuous evaluation run. Requires --continuous-session-dir for stage 2/3.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "reports" / "realtime_tests"),
        help="Realtime session output root for continuous evaluation.",
    )
    parser.add_argument(
        "--sentence-manifest",
        default="",
        help="Optional sentence manifest for frame-level reference labels.",
    )
    parser.add_argument(
        "--segments-manifest",
        default="",
        help="Optional sentence word-segment manifest for frame-level reference labels.",
    )
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--mirror-input", action="store_true", default=True)
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional Markdown report output path. Defaults to <artifact-dir>/models/post_train_evaluation_summary.md.",
    )
    parser.add_argument(
        "--engine-mode",
        choices=("sliding_window", "trigger_based"),
        default="sliding_window",
        help="Inference engine mode for continuous evaluation.",
    )
    return parser


def _canonical_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _derive_dataset_dir(artifact_dir: Path) -> Path:
    name = artifact_dir.name
    if name.startswith("artifacts_"):
        return artifact_dir.parent / name.replace("artifacts_", "dataset_pipeline_", 1) / "processed_sequences"
    return ROOT / "dataset_pipeline" / "processed_sequences"


def _derive_split_csv(data_dir: Path) -> Path:
    return data_dir.parent / "splits" / "test.csv"


def _auto_select_scripted_video(scripted_glob: str) -> Path | None:
    candidates = sorted(path.resolve() for path in PROJECT_ROOT.glob(scripted_glob) if path.exists())
    return candidates[0] if candidates else None


def _auto_select_continuous_video(sentence_manifest: Path) -> Path | None:
    if not sentence_manifest.exists():
        return None
    rows = _load_csv_rows(sentence_manifest)
    preferred_row = None
    fallback_row = None
    for row in rows:
        output_video = Path(row["output_video"])
        if not output_video.exists():
            continue
        tokens = [token.strip().lower() for token in json.loads(row["tokens_json"])]
        if any(label in tokens for label in HARDCASE_POSITION_LABELS):
            if all(label in tokens for label in HARDCASE_POSITION_LABELS):
                preferred_row = row
                break
            if fallback_row is None:
                fallback_row = row
    first_existing = next((row for row in rows if Path(row["output_video"]).exists()), None)
    chosen = preferred_row or fallback_row or first_existing
    if chosen is None:
        return None
    return Path(chosen["output_video"])


def _ensure_offline_outputs(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    test_split_csv: Path,
    test_eval_json: Path,
    test_predictions_csv: Path,
    force: bool,
) -> None:
    if not force and test_eval_json.exists() and test_predictions_csv.exists():
        return
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_multibranch.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--data-dir",
        str(data_dir),
        "--split-csv",
        str(test_split_csv),
        "--output-json",
        str(test_eval_json),
        "--output-csv",
        str(test_predictions_csv),
    ]
    subprocess.run(command, check=True, cwd=str(ROOT))


def _build_confusion(prediction_rows: Iterable[dict[str, str]]) -> tuple[list[str], dict[str, dict[str, int]]]:
    labels = sorted(
        {
            row["true_label"].strip().lower()
            for row in prediction_rows
        }
        | {
            row["pred_label"].strip().lower()
            for row in prediction_rows
        }
    )
    confusion: dict[str, dict[str, int]] = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }
    for row in prediction_rows:
        true_label = row["true_label"].strip().lower()
        pred_label = row["pred_label"].strip().lower()
        confusion[true_label][pred_label] += 1
    return labels, confusion


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _compute_class_metrics(labels: list[str], confusion: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label].values())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
            "accuracy": recall,
        }
    return metrics


def _group_accuracy(group_labels: tuple[str, ...], confusion: dict[str, dict[str, int]]) -> float:
    total = sum(sum(confusion[label].values()) for label in group_labels if label in confusion)
    correct = sum(confusion[label].get(label, 0) for label in group_labels if label in confusion)
    return _safe_div(correct, total)


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    rendered_rows = [[str(cell) for cell in row] for row in rows]
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rendered_rows) if rendered_rows else ""
    return "\n".join(part for part in [header_line, divider_line, body] if part)


def _render_group_metric_table(group_labels: tuple[str, ...], metrics: dict[str, dict[str, float]]) -> str:
    rows = []
    for label in group_labels:
        data = metrics.get(label, {})
        rows.append(
            [
                label,
                _format_pct(float(data.get("precision", 0.0))),
                _format_pct(float(data.get("recall", 0.0))),
                _format_pct(float(data.get("f1", 0.0))),
                int(data.get("support", 0.0)),
            ]
        )
    return _markdown_table(["Label", "Precision", "Recall", "F1", "Support"], rows)


def _render_group_confusion_table(group_labels: tuple[str, ...], confusion: dict[str, dict[str, int]]) -> str:
    headers = ["True \\ Pred", *group_labels]
    rows = []
    for true_label in group_labels:
        rows.append([true_label, *[confusion.get(true_label, {}).get(pred_label, 0) for pred_label in group_labels]])
    return _markdown_table(headers, rows)


def _render_pair_confusion_lines(group_labels: tuple[str, ...], confusion: dict[str, dict[str, int]]) -> list[str]:
    lines = []
    for true_label in group_labels:
        for pred_label in group_labels:
            if true_label == pred_label:
                continue
            lines.append(f"- `{true_label} -> {pred_label}`: {confusion.get(true_label, {}).get(pred_label, 0)}")
    return lines


def _frame_rows_to_signals(frame_rows: list[dict[str, str]]) -> list[ScriptedFrameSignal]:
    signals = []
    for index, row in enumerate(frame_rows):
        signals.append(
            ScriptedFrameSignal(
                frame_index=int(row.get("frame_index") or index),
                signal_score=float(row.get("signal_score") or 0.0),
                motion_energy=float(row.get("motion_energy") or 0.0),
                status=str(row.get("status", "")),
                left_hand_present=str(row.get("left_hand_present", "0")).strip() in {"1", "true", "True"},
                right_hand_present=str(row.get("right_hand_present", "0")).strip() in {"1", "true", "True"},
                pose_present=str(row.get("pose_present", "0")).strip() in {"1", "true", "True"},
            )
        )
    return signals


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

    expected_tokens = [token.strip().lower() for token in json.loads(sentence_row["tokens_json"])]
    frame_labels = ["no_sign" for _ in range(frame_count)]
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


def _load_scripted_reference(
    video_path: Path,
    frame_rows: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    frame_signals = _frame_rows_to_signals(frame_rows)
    expected_tokens = parse_script_tokens_from_path(video_path)
    if not expected_tokens:
        raise FileNotFoundError(f"No scripted tokens parsed from filename: {video_path}")
    alignment = build_scripted_alignment(video_path, frame_signals, allowed_labels=set(expected_tokens))
    if alignment is None:
        raise FileNotFoundError(f"Unable to build scripted alignment for: {video_path}")
    frame_labels = [CONFIG.no_sign_label for _ in range(len(frame_rows))]
    for segment in alignment.get("segments", []):
        token = str(segment["token"]).strip().lower()
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        for frame_index in range(max(0, start), min(len(frame_labels), end + 1)):
            frame_labels[frame_index] = token
    return [token.strip().lower() for token in alignment.get("expected_tokens", expected_tokens)], frame_labels


def _run_continuous_evaluation(
    *,
    checkpoint_path: Path,
    video_path: Path,
    output_root: Path,
    sentence_manifest: Path,
    segments_manifest: Path,
    scripted_glob: str,
    sequence_length: int,
    confidence_threshold: float,
    mirror_input: bool,
    engine_mode: str,
) -> Path:
    before = {
        item.resolve()
        for item in output_root.glob("realtime_test_*")
        if item.is_dir()
    }
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_scripted_video_realtime_regression.py"),
        "--mode",
        "continuous",
        "--engine-mode",
        str(engine_mode),
        "--video",
        str(video_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output-root",
        str(output_root),
        "--scripted-glob",
        str(scripted_glob),
        "--sequence-length",
        str(sequence_length),
        "--confidence-threshold",
        str(confidence_threshold),
    ]
    if sentence_manifest.exists():
        command.extend(["--sentence-manifest", str(sentence_manifest)])
    if segments_manifest.exists():
        command.extend(["--segments-manifest", str(segments_manifest)])
    if mirror_input:
        command.append("--mirror-input")
    subprocess.run(command, check=True, cwd=str(ROOT))
    after = [
        item.resolve()
        for item in output_root.glob("realtime_test_*")
        if item.is_dir()
    ]
    new_dirs = sorted(path for path in after if path not in before)
    if new_dirs:
        return new_dirs[-1]
    if after:
        return sorted(after, key=lambda path: path.stat().st_mtime)[-1]
    raise FileNotFoundError(f"No realtime session directory found under: {output_root}")


def _analyze_disambiguation(
    frame_rows: list[dict[str, str]],
    reference_frame_labels: list[str],
) -> dict[str, dict[str, float]]:
    per_label: dict[str, dict[str, float]] = {}
    for label in HARDCASE_POSITION_LABELS:
        applied_rows = []
        for index, row in enumerate(frame_rows):
            if index >= len(reference_frame_labels):
                break
            raw_label = row["raw_label"].strip().lower()
            final_label = row["final_label"].strip().lower()
            reference_label = reference_frame_labels[index].strip().lower()
            applied = str(row.get("disambiguation_applied", "0")).strip() in {"1", "true", "True"}
            if not applied:
                continue
            if label not in {raw_label, final_label, reference_label}:
                continue
            applied_rows.append((raw_label, final_label, reference_label))

        applied_count = len(applied_rows)
        corrected = sum(1 for raw_label, final_label, reference_label in applied_rows if raw_label != reference_label and final_label == reference_label)
        harmful = sum(1 for raw_label, final_label, reference_label in applied_rows if raw_label == reference_label and final_label != reference_label)
        kept_correct = sum(1 for raw_label, final_label, reference_label in applied_rows if raw_label == reference_label and final_label == reference_label)
        still_wrong = sum(1 for raw_label, final_label, reference_label in applied_rows if raw_label != reference_label and final_label != reference_label)
        per_label[label] = {
            "applied_count": float(applied_count),
            "corrected_count": float(corrected),
            "harmful_count": float(harmful),
            "kept_correct_count": float(kept_correct),
            "still_wrong_count": float(still_wrong),
            "success_rate": _safe_div(corrected, applied_count),
        }
    return per_label


def _render_disambiguation_table(stats: dict[str, dict[str, float]]) -> str:
    rows = []
    for label in HARDCASE_POSITION_LABELS:
        data = stats.get(label, {})
        rows.append(
            [
                label,
                int(data.get("applied_count", 0.0)),
                int(data.get("corrected_count", 0.0)),
                int(data.get("kept_correct_count", 0.0)),
                int(data.get("harmful_count", 0.0)),
                int(data.get("still_wrong_count", 0.0)),
                _format_pct(float(data.get("success_rate", 0.0))),
            ]
        )
    return _markdown_table(
        ["Label", "Applied", "Corrected", "Kept Correct", "Harmful", "Still Wrong", "Success Rate"],
        rows,
    )


def _build_report(
    *,
    artifact_dir: Path,
    checkpoint_path: Path,
    test_eval_json: Path,
    test_predictions_csv: Path,
    continuous_session_dir: Path,
    continuous_video_path: Path,
    offline_metrics: dict[str, dict[str, float]],
    confusion: dict[str, dict[str, int]],
    continuous_summary: dict[str, object],
    disambiguation_stats: dict[str, dict[str, float]],
) -> str:
    continuous_eval = continuous_summary.get("continuous_evaluation", {})
    per_label_frame_metrics = continuous_eval.get("per_label_frame_metrics", {})
    no_sign_metrics = per_label_frame_metrics.get("no_sign", {})

    sections: list[str] = []
    sections.append("# Post-Train Evaluation Summary")
    sections.append("")
    sections.append("## Artifact")
    sections.append(f"- Artifact dir: `{artifact_dir}`")
    sections.append(f"- Checkpoint: `{checkpoint_path}`")
    sections.append(f"- Offline eval JSON: `{test_eval_json}`")
    sections.append(f"- Offline predictions CSV: `{test_predictions_csv}`")
    sections.append(f"- Continuous session: `{continuous_session_dir}`")
    sections.append(f"- Continuous video: `{continuous_video_path}`")
    sections.append("")
    sections.append("## Offline Confusion Check")
    sections.append("")
    sections.append("### Fine Handshape Group: `i`, `like`, `you`")
    sections.append(_render_group_metric_table(HARDCASE_FINGER_LABELS, offline_metrics))
    sections.append("")
    sections.append("Cross-confusion counts:")
    sections.extend(_render_pair_confusion_lines(HARDCASE_FINGER_LABELS, confusion))
    sections.append("")
    sections.append(_render_group_confusion_table(HARDCASE_FINGER_LABELS, confusion))
    sections.append("")
    sections.append("### Same Handshape, Different Position: `father`, `mother`")
    sections.append(_render_group_metric_table(HARDCASE_POSITION_LABELS, offline_metrics))
    sections.append("")
    sections.append(f"Pair accuracy: `{_format_pct(_group_accuracy(HARDCASE_POSITION_LABELS, confusion))}`")
    sections.append("Cross-confusion counts:")
    sections.extend(_render_pair_confusion_lines(HARDCASE_POSITION_LABELS, confusion))
    sections.append("")
    sections.append(_render_group_confusion_table(HARDCASE_POSITION_LABELS, confusion))
    sections.append("")
    sections.append("## Disambiguation Trigger Analysis")
    sections.append("")
    sections.append(_render_disambiguation_table(disambiguation_stats))
    sections.append("")
    sections.append("Success rate here is defined as `raw wrong -> final corrected` after the rule fired.")
    sections.append("")
    sections.append("## Continuous Stability")
    sections.append("")
    sections.append(f"- Token-level WER: `{float(continuous_eval.get('word_error_rate', 0.0)):.4f}`")
    sections.append(f"- Frame macro F1: `{float(continuous_eval.get('frame_macro_f1', 0.0)):.4f}`")
    sections.append(f"- `no_sign` frame F1: `{float(no_sign_metrics.get('f1', 0.0)):.4f}`")
    sections.append(f"- `no_sign` recall: `{float(continuous_eval.get('no_sign_recall', 0.0)):.4f}`")
    sections.append(f"- Probability timeline: `{continuous_eval.get('label_probability_plot', '')}`")
    trigger_segment_report = continuous_eval.get("trigger_segment_report", {}) or {}
    if trigger_segment_report:
        sections.append(f"- Trigger segment alignment: `{trigger_segment_report.get('alignment_csv', '')}`")
        sections.append(f"- Trigger segment confusion: `{trigger_segment_report.get('confusion_json', '')}`")
    sections.append("")
    sections.append("## Interpretation")
    sections.append("")
    sections.append(
        f"- `i/like/you` should be judged mainly from the first table plus their mutual confusion counts; lower off-diagonal counts mean the explicit finger features are carrying more signal."
    )
    sections.append(
        f"- `father/mother` should be judged by combining the offline pair accuracy with the trigger table; offline reflects the base model, while trigger success reflects the rule-based correction in continuous decoding."
    )
    sections.append(
        f"- Transition stability should be judged from WER together with `no_sign` frame F1 and recall; higher `no_sign` scores usually mean less transition jitter."
    )
    sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else (artifact_dir / "models" / "multibranch_baseline.pt").resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_dir = Path(args.data_dir).resolve() if args.data_dir else _derive_dataset_dir(artifact_dir).resolve()
    test_split_csv = Path(args.test_split_csv).resolve() if args.test_split_csv else _derive_split_csv(data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed sequences directory not found: {data_dir}")
    if not test_split_csv.exists():
        raise FileNotFoundError(f"Test split CSV not found: {test_split_csv}")

    models_dir = checkpoint_path.parent
    test_eval_json = Path(args.test_eval_json).resolve() if args.test_eval_json else (models_dir / "test_eval.json").resolve()
    test_predictions_csv = (
        Path(args.test_predictions_csv).resolve()
        if args.test_predictions_csv
        else (models_dir / "test_predictions.csv").resolve()
    )
    _ensure_offline_outputs(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        test_split_csv=test_split_csv,
        test_eval_json=test_eval_json,
        test_predictions_csv=test_predictions_csv,
        force=bool(args.force_test_inference),
    )

    prediction_rows = _load_csv_rows(test_predictions_csv)
    labels, confusion = _build_confusion(prediction_rows)
    offline_metrics = _compute_class_metrics(labels, confusion)

    sentence_manifest = (
        Path(args.sentence_manifest).resolve()
        if args.sentence_manifest
        else (PROJECT_ROOT / "metadata" / "scripted_sentence_manifest.csv").resolve()
    )
    segments_manifest = (
        Path(args.segments_manifest).resolve()
        if args.segments_manifest
        else (PROJECT_ROOT / "metadata" / "scripted_word_segments_manifest.csv").resolve()
    )
    output_root = Path(args.output_root).resolve()

    if args.continuous_session_dir:
        continuous_session_dir = Path(args.continuous_session_dir).resolve()
        if not continuous_session_dir.exists():
            raise FileNotFoundError(f"Continuous session directory not found: {continuous_session_dir}")
        session_summary = json.loads((continuous_session_dir / "session_summary.json").read_text(encoding="utf-8"))
        continuous_video_path = Path(session_summary.get("source_video_path", "")).resolve()
    else:
        if args.skip_continuous_run:
            raise ValueError("--skip-continuous-run requires --continuous-session-dir.")
        if args.continuous_video:
            continuous_video_path = Path(args.continuous_video).resolve()
        else:
            continuous_video_path = _auto_select_scripted_video(args.scripted_glob)
            if continuous_video_path is None:
                continuous_video_path = _auto_select_continuous_video(sentence_manifest)
            if continuous_video_path is None:
                raise FileNotFoundError(
                    "No usable continuous evaluation video found. "
                    "Pass --continuous-video explicitly or provide valid sentence manifests."
                )
        continuous_session_dir = _run_continuous_evaluation(
            checkpoint_path=checkpoint_path,
            video_path=continuous_video_path,
            output_root=output_root,
            sentence_manifest=sentence_manifest,
            segments_manifest=segments_manifest,
            scripted_glob=args.scripted_glob,
            sequence_length=int(args.sequence_length),
            confidence_threshold=float(args.confidence_threshold),
            mirror_input=bool(args.mirror_input),
            engine_mode=str(args.engine_mode),
        )
        session_summary = json.loads((continuous_session_dir / "session_summary.json").read_text(encoding="utf-8"))

    frame_predictions_csv = continuous_session_dir / "frame_predictions.csv"
    if not frame_predictions_csv.exists():
        raise FileNotFoundError(f"frame_predictions.csv not found: {frame_predictions_csv}")
    frame_rows = _load_csv_rows(frame_predictions_csv)
    source_video_path = Path(session_summary.get("source_video_path") or continuous_video_path).resolve()
    try:
        _, reference_frame_labels = _load_sentence_reference(
            source_video_path,
            sentence_manifest,
            segments_manifest,
            frame_count=len(frame_rows),
        )
    except FileNotFoundError:
        _, reference_frame_labels = _load_scripted_reference(source_video_path, frame_rows)
    disambiguation_stats = _analyze_disambiguation(frame_rows, reference_frame_labels)

    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else (models_dir / "post_train_evaluation_summary.md").resolve()
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = _build_report(
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        test_eval_json=test_eval_json,
        test_predictions_csv=test_predictions_csv,
        continuous_session_dir=continuous_session_dir,
        continuous_video_path=source_video_path,
        offline_metrics=offline_metrics,
        confusion=confusion,
        continuous_summary=session_summary,
        disambiguation_stats=disambiguation_stats,
    )
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
