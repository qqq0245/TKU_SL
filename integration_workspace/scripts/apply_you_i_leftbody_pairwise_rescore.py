from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.continuous_feature_cache import load_continuous_feature_cache
from src.models.feature_group_transforms import apply_hand_mask_validity_scale, apply_pose_hip_coordinate_scale
from src.models.feature_slices import split_feature_tensor
from src.models.inference_utils_multibranch import load_multibranch_checkpoint


LEFT_BODY_START = 214 + 9
LEFT_BODY_END = LEFT_BODY_START + 9
LEFT_BODY_CHEST_Y_OFFSET = 4
LEFT_BODY_TORSO_Y_OFFSET = 7
DEFAULT_Y_OFFSET = -0.035


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Apply a residual-specific you-vs-i pairwise left-body logit correction.")
    parser.add_argument("--classification-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--mirror-input", type=int, default=1)
    parser.add_argument("--left-body-y-offset", type=float, default=DEFAULT_Y_OFFSET)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: float) -> float:
    return round(float(value), 6)


def _probabilities_from_logits(logits_np: np.ndarray, index_to_label: dict[int, str]) -> tuple[list[dict[str, object]], dict[str, float]]:
    stable = logits_np.astype(np.float64)
    stable -= np.max(stable)
    probs = np.exp(stable)
    probs /= max(float(np.sum(probs)), 1e-12)
    label_scores = {
        str(index_to_label[int(index)]): float(probs[int(index)])
        for index in range(len(probs))
    }
    top_indices = np.argsort(-probs)[:3]
    top3: list[dict[str, object]] = []
    for index in top_indices:
        index_value = int(index)
        top3.append(
            {
                "label": str(index_to_label[index_value]),
                "confidence": _safe_float(probs[index_value]),
                "logit": _safe_float(logits_np[index_value]),
            }
        )
    return top3, label_scores


def _predict_logits(
    *,
    model,
    sequence: np.ndarray,
    feature_mode: str,
    feature_spec: dict | None,
    hand_mask_validity_scale: float,
    pose_hip_coordinate_scale: float,
) -> np.ndarray:
    adjusted_sequence = apply_pose_hip_coordinate_scale(
        sequence,
        feature_spec=feature_spec,
        scale=pose_hip_coordinate_scale,
    )
    adjusted_sequence = apply_hand_mask_validity_scale(
        adjusted_sequence,
        feature_spec=feature_spec,
        scale=hand_mask_validity_scale,
    )
    tensor = torch.from_numpy(adjusted_sequence).unsqueeze(0)
    parts = split_feature_tensor(tensor, feature_mode, feature_spec=feature_spec)
    with torch.no_grad():
        logits = model(parts["skeleton_stream"], parts["location_stream"], parts["motion_stream"])
    return logits.squeeze(0).detach().cpu().numpy()


def _rebuild_pose_local_sequence(
    *,
    base_sequence: np.ndarray,
    arrays: dict[str, np.ndarray],
    sampled_indices: np.ndarray,
    pose_local_anchor: str,
) -> np.ndarray:
    anchor_mode = str(pose_local_anchor).strip().lower()
    if anchor_mode == "mid_shoulder":
        return base_sequence
    if anchor_mode != "torso_center":
        raise ValueError(f"Unsupported pose-local anchor: {pose_local_anchor}")
    adjusted = base_sequence.copy()
    channels = 3
    left_xyz = 21 * channels
    right_xyz = 21 * channels
    pose_coord_start = left_xyz + right_xyz
    pose_coord_end = pose_coord_start + (9 * channels)
    for row_index, frame_index in enumerate(sampled_indices.astype(int).tolist()):
        adjusted[row_index, pose_coord_start:pose_coord_end] = arrays["normalized_pose"][frame_index].reshape(-1).astype(
            np.float32
        )
    return adjusted


def _distance_gap(vector: np.ndarray, true_proto: np.ndarray, confused_proto: np.ndarray) -> float:
    return float(np.linalg.norm(vector - true_proto) - np.linalg.norm(vector - confused_proto))


def _update_summary(payload: dict[str, object], mirror_input: int) -> dict[str, object]:
    updated_payload = deepcopy(payload)
    rows = list(updated_payload.get("classifications", []))
    updated_payload["classifications"] = rows
    summary_rows = list(updated_payload.get("summary_by_mirror", []))
    for summary in summary_rows:
        if int(summary.get("mirror_input", -1)) != mirror_input:
            continue
        mirror_rows = [row for row in rows if int(row.get("mirror_input", -1)) == mirror_input]
        predicted_tokens = [str(row["predicted_label"]) for row in mirror_rows]
        reference_tokens = [str(row["token"]) for row in mirror_rows]
        correct_count = sum(
            int(str(row["predicted_label"]).strip().lower() == str(row["token"]).strip().lower())
            for row in mirror_rows
        )
        confusion: dict[str, dict[str, int]] = {}
        for row in mirror_rows:
            reference = str(row["token"])
            predicted = str(row["predicted_label"])
            confusion.setdefault(reference, {})
            confusion[reference][predicted] = confusion[reference].get(predicted, 0) + 1
        summary["correct_count"] = correct_count
        summary["exact_span_accuracy"] = _safe_float(correct_count / max(1, len(mirror_rows)))
        summary["predicted_tokens"] = predicted_tokens
        summary["predicted_label_counts"] = dict(Counter(predicted_tokens))
        summary["confusion"] = confusion
        summary["reference_tokens"] = reference_tokens
    updated_payload["summary_by_mirror"] = summary_rows
    return updated_payload


def main() -> None:
    args = build_parser().parse_args()
    classification_path = Path(args.classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    analysis_path = Path(args.analysis_json).resolve()
    output_path = Path(args.output_json).resolve()
    summary_path = Path(args.summary_json).resolve()
    mirror_input = int(args.mirror_input)
    left_body_y_offset = float(args.left_body_y_offset)

    classification_payload = _load_json(classification_path)
    analysis_payload = _load_json(analysis_path)
    dominant_group = str(analysis_payload.get("dominant_nonface_subgroup", ""))
    if dominant_group != "left_body_vectors":
        raise RuntimeError(f"Expected dominant_nonface_subgroup=left_body_vectors, got {dominant_group!r}")

    cache_path = cache_dir / f"continuous_feature_cache_mirror{mirror_input}.npz"
    _, arrays = load_continuous_feature_cache(cache_path)
    feature_vectors = arrays["feature_vectors"].astype(np.float32)

    device = torch.device("cpu")
    model, checkpoint = load_multibranch_checkpoint(checkpoint_path, device)
    if model is None or checkpoint is None:
        raise RuntimeError(f"Unable to load checkpoint: {checkpoint_path}")
    model.eval()
    feature_mode = str(checkpoint["feature_mode"])
    feature_spec = checkpoint.get("feature_spec")
    index_to_label = {int(index): str(label) for index, label in checkpoint["index_to_label"].items()}
    label_to_index = {label: index for index, label in index_to_label.items()}

    baseline_summary = deepcopy(
        next(
            row for row in classification_payload.get("summary_by_mirror", [])
            if int(row.get("mirror_input", -1)) == mirror_input
        )
    )

    dominant_rows = analysis_payload.get("dominant_subgroup_column_breakdown", [])
    name_to_row = {str(row["name"]): row for row in dominant_rows}
    column_names = [
        "left_to_shoulder_center_x",
        "left_to_shoulder_center_y",
        "left_to_shoulder_center_z",
        "left_to_chest_center_x",
        "left_to_chest_center_y",
        "left_to_chest_center_z",
        "left_to_torso_center_x",
        "left_to_torso_center_y",
        "left_to_torso_center_z",
    ]
    you_proto = np.asarray([float(name_to_row[name]["true_prototype_mean"]) for name in column_names], dtype=np.float32)
    i_proto = np.asarray([float(name_to_row[name]["confused_prototype_mean"]) for name in column_names], dtype=np.float32)

    updated_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for row in classification_payload.get("classifications", []):
        updated = dict(row)
        if int(row.get("mirror_input", -1)) != mirror_input:
            updated_rows.append(updated)
            continue

        sampled_indices = np.asarray(row.get("sampled_frame_indices", []), dtype=np.int32)
        if sampled_indices.size == 0:
            updated_rows.append(updated)
            continue
        sequence = _rebuild_pose_local_sequence(
            base_sequence=feature_vectors[sampled_indices].astype(np.float32),
            arrays=arrays,
            sampled_indices=sampled_indices,
            pose_local_anchor=str(classification_payload.get("pose_local_anchor", "")),
        )
        left_body_mean = sequence[:, LEFT_BODY_START:LEFT_BODY_END].mean(axis=0)
        baseline_gap = _distance_gap(left_body_mean, you_proto, i_proto)
        adjusted_mean = left_body_mean.copy()
        adjusted_mean[LEFT_BODY_CHEST_Y_OFFSET] += left_body_y_offset
        adjusted_mean[LEFT_BODY_TORSO_Y_OFFSET] += left_body_y_offset
        adjusted_gap = _distance_gap(adjusted_mean, you_proto, i_proto)

        baseline_logits = _predict_logits(
            model=model,
            sequence=sequence,
            feature_mode=feature_mode,
            feature_spec=feature_spec,
            hand_mask_validity_scale=float(classification_payload.get("hand_mask_validity_scale", 1.0)),
            pose_hip_coordinate_scale=float(classification_payload.get("pose_hip_coordinate_scale", 1.0)),
        )
        baseline_top3, _baseline_probabilities = _probabilities_from_logits(baseline_logits, index_to_label)
        you_index = label_to_index["you"]
        i_index = label_to_index["i"]
        should_apply = bool(
            str(row.get("predicted_label", "")).strip().lower() == "i"
            and baseline_gap > 0.0
            and adjusted_gap < 0.0
        )

        candidate_info = {
            "token": str(row["token"]),
            "baseline_prediction": str(row.get("predicted_label", "")),
            "baseline_top3": list(row.get("top3", [])),
            "baseline_left_body_you_minus_i_l2": _safe_float(baseline_gap),
            "adjusted_left_body_you_minus_i_l2": _safe_float(adjusted_gap),
            "left_body_y_offset": _safe_float(left_body_y_offset),
            "applied": should_apply,
        }

        if should_apply:
            calibration_gain = max(baseline_gap - adjusted_gap, 0.0)
            updated_logits = baseline_logits.copy()
            you_logit = float(updated_logits[you_index])
            i_logit = float(updated_logits[i_index])
            updated_logits[you_index] = max(you_logit, i_logit) + calibration_gain
            updated_logits[i_index] = min(you_logit, i_logit)
            adjusted_top3, adjusted_probabilities = _probabilities_from_logits(updated_logits, index_to_label)
            updated["top3"] = adjusted_top3
            updated["predicted_label"] = str(adjusted_top3[0]["label"]) if adjusted_top3 else str(updated.get("predicted_label", ""))
            updated["top1_logit"] = _safe_float(updated_logits[int(np.argmax(updated_logits))])
            updated["correct"] = int(str(updated["predicted_label"]).strip().lower() == str(updated["token"]).strip().lower())
            updated["you_i_leftbody_pairwise_rescore"] = {
                "left_body_y_offset": _safe_float(left_body_y_offset),
                "calibrated_columns": ["left_to_chest_center_y", "left_to_torso_center_y"],
                "baseline_left_body_you_minus_i_l2": _safe_float(baseline_gap),
                "adjusted_left_body_you_minus_i_l2": _safe_float(adjusted_gap),
                "pairwise_logit_gain": _safe_float(calibration_gain),
                "baseline_you_logit": _safe_float(you_logit),
                "baseline_i_logit": _safe_float(i_logit),
                "adjusted_top3": adjusted_top3,
                "adjusted_you_probability": _safe_float(adjusted_probabilities.get("you", 0.0)),
                "adjusted_i_probability": _safe_float(adjusted_probabilities.get("i", 0.0)),
            }
            candidate_info["pairwise_logit_gain"] = _safe_float(calibration_gain)
            candidate_info["adjusted_top3"] = adjusted_top3
        else:
            updated["correct"] = int(str(updated["predicted_label"]).strip().lower() == str(updated["token"]).strip().lower())

        updated_rows.append(updated)
        if str(row.get("predicted_label", "")).strip().lower() == "i":
            candidate_rows.append(candidate_info)

    updated_payload = deepcopy(classification_payload)
    updated_payload["classifications"] = updated_rows
    updated_payload["you_i_leftbody_pairwise_rescore"] = {
        "analysis_json": str(analysis_path),
        "left_body_y_offset": _safe_float(left_body_y_offset),
        "calibrated_columns": ["left_to_chest_center_y", "left_to_torso_center_y"],
        "rescore_scope": "only spans whose baseline top1 is i and whose left_body you-vs-i distance flips sign after calibration",
        "candidate_rows": candidate_rows,
    }
    updated_payload = _update_summary(updated_payload, mirror_input)
    output_path.write_text(json.dumps(updated_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    updated_summary = next(
        row for row in updated_payload.get("summary_by_mirror", [])
        if int(row.get("mirror_input", -1)) == mirror_input
    )
    you_row = next(
        row for row in updated_rows
        if int(row.get("mirror_input", -1)) == mirror_input and str(row.get("token", "")).strip().lower() == "you"
    )
    mother_row = next(
        row for row in updated_rows
        if int(row.get("mirror_input", -1)) == mirror_input and str(row.get("token", "")).strip().lower() == "mother"
    )

    summary_payload = {
        "baseline_classification_json": str(classification_path),
        "analysis_json": str(analysis_path),
        "output_classification_json": str(output_path),
        "left_body_y_offset": _safe_float(left_body_y_offset),
        "calibrated_columns": ["left_to_chest_center_y", "left_to_torso_center_y"],
        "baseline_accuracy": baseline_summary["exact_span_accuracy"],
        "updated_accuracy": updated_summary["exact_span_accuracy"],
        "baseline_predicted_label_counts": baseline_summary["predicted_label_counts"],
        "updated_predicted_label_counts": updated_summary["predicted_label_counts"],
        "baseline_confusion": baseline_summary["confusion"],
        "updated_confusion": updated_summary["confusion"],
        "candidate_rows": candidate_rows,
        "improved": bool(updated_summary["exact_span_accuracy"] > baseline_summary["exact_span_accuracy"]),
        "you_confusion_weakened": bool(
            baseline_summary["confusion"].get("you", {}).get("i", 0) > 0
            and updated_summary["confusion"].get("you", {}).get("i", 0) == 0
            and updated_summary["confusion"].get("you", {}).get("you", 0) > 0
        ),
        "mother_stable": bool(str(mother_row.get("predicted_label", "")).strip().lower() == "you"),
        "you_prediction_after_fix": str(you_row.get("predicted_label", "")),
        "you_top3_after_fix": list(you_row.get("top3", [])),
        "pairwise_scope": "you/i only",
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")
    print(f"[OK] Wrote {summary_path}")


if __name__ == "__main__":
    main()
