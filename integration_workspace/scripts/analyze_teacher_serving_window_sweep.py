from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
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


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Sweep teacher serving windows across anchors.")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--exact-classification-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "artifacts_webcam9_relative_coord_v1" / "models" / "multibranch_baseline.pt"),
    )
    parser.add_argument("--output-json", required=True)
    return parser


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _rebuild_pose_local_sequence(
    *,
    base_sequence: np.ndarray,
    arrays: dict[str, np.ndarray],
    sampled_indices: np.ndarray,
    feature_spec: dict,
    pose_local_anchor: str,
) -> np.ndarray:
    anchor_mode = str(pose_local_anchor).strip().lower()
    if anchor_mode == "mid_shoulder":
        return base_sequence
    if anchor_mode != "torso_center":
        raise ValueError(f"Unsupported pose anchor: {pose_local_anchor}")
    adjusted = base_sequence.copy()
    pose_coord_start = (21 * 3) + (21 * 3)
    pose_coord_end = pose_coord_start + (9 * 3)
    for row_index, frame_index in enumerate(sampled_indices.astype(int).tolist()):
        adjusted[row_index, pose_coord_start:pose_coord_end] = arrays["normalized_pose"][frame_index].reshape(-1).astype(np.float32)
    return adjusted


@torch.no_grad()
def _score_sequence(
    *,
    model,
    feature_mode: str,
    feature_spec: dict,
    index_to_label: dict[int, str],
    sequence: np.ndarray,
) -> dict[str, object]:
    adjusted = apply_pose_hip_coordinate_scale(sequence, feature_spec=feature_spec, scale=1.0)
    adjusted = apply_hand_mask_validity_scale(adjusted, feature_spec=feature_spec, scale=1.0)
    tensor = torch.from_numpy(adjusted).unsqueeze(0)
    parts = split_feature_tensor(tensor, feature_mode, feature_spec=feature_spec)
    logits = model(parts["skeleton_stream"], parts["location_stream"], parts["motion_stream"]).squeeze(0).detach().cpu().numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=0).detach().cpu().numpy()
    label_to_prob = {index_to_label[int(index)]: float(probabilities[int(index)]) for index in range(len(probabilities))}
    label_to_logit = {index_to_label[int(index)]: float(logits[int(index)]) for index in range(len(logits))}
    top_indices = np.argsort(-probabilities)[:3]
    top3 = [
        {
            "label": index_to_label[int(index)],
            "confidence": round(float(probabilities[int(index)]), 6),
            "logit": round(float(logits[int(index)]), 6),
        }
        for index in top_indices
    ]
    teacher_logit = label_to_logit.get("teacher", -1e9)
    student_logit = label_to_logit.get("student", -1e9)
    no_sign_logit = label_to_logit.get("no_sign", -1e9)
    return {
        "top3": top3,
        "predicted_label": top3[0]["label"] if top3 else "",
        "teacher_probability": round(float(label_to_prob.get("teacher", 0.0)), 6),
        "student_probability": round(float(label_to_prob.get("student", 0.0)), 6),
        "no_sign_probability": round(float(label_to_prob.get("no_sign", 0.0)), 6),
        "teacher_minus_student_logit_gap": round(float(teacher_logit - student_logit), 6),
        "teacher_minus_nosign_logit_gap": round(float(teacher_logit - no_sign_logit), 6),
    }


def _variant_indices(start_frame: int, end_frame: int, variant: str) -> np.ndarray:
    length = max(end_frame - start_frame + 1, 1)
    if variant == "uniform_full":
        return np.linspace(start_frame, end_frame, 30).round().astype(int)
    if length <= 30:
        return np.linspace(start_frame, end_frame, 30).round().astype(int)
    if variant == "first_30":
        return np.arange(start_frame, start_frame + 30, dtype=int)
    if variant == "last_30":
        return np.arange(end_frame - 29, end_frame + 1, dtype=int)
    if variant == "middle_30":
        middle_start = start_frame + max((length - 30) // 2, 0)
        return np.arange(middle_start, middle_start + 30, dtype=int)
    raise ValueError(f"Unsupported variant: {variant}")


def main() -> None:
    args = build_parser().parse_args()
    session_dir = Path(args.session_dir).resolve()
    exact_json = Path(args.exact_classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_json = Path(args.output_json).resolve()

    exact_payload = json.loads(exact_json.read_text(encoding="utf-8"))
    exact_teacher = next(
        row
        for row in exact_payload["classifications"]
        if int(row.get("mirror_input", 0)) == 1 and str(row.get("token", "")).strip().lower() == "teacher"
    )
    alignment_rows = _load_csv(session_dir / "trigger_segment_alignment.csv")
    teacher_emit = next(row for row in alignment_rows if row.get("segment_id") == "7")
    student_reject = next(row for row in alignment_rows if row.get("segment_id") == "8")

    metadata, arrays = load_continuous_feature_cache(cache_dir / "continuous_feature_cache_mirror1.npz")
    model, checkpoint = load_multibranch_checkpoint(Path(args.checkpoint).resolve(), torch.device("cpu"))
    if model is None or checkpoint is None:
        raise RuntimeError(f"Unable to load checkpoint: {args.checkpoint}")
    model.eval()
    feature_mode = str(checkpoint["feature_mode"])
    feature_spec = checkpoint.get("feature_spec")
    index_to_label = checkpoint["index_to_label"]

    windows = [
        {
            "name": "teacher_exact_full",
            "start_frame": int(exact_teacher["start_frame"]),
            "end_frame": int(exact_teacher["end_frame"]),
            "variants": ["uniform_full", "first_30", "middle_30", "last_30"],
        },
        {
            "name": "teacher_serving_emit",
            "start_frame": int(teacher_emit["start_frame"]),
            "end_frame": int(teacher_emit["end_frame"]),
            "variants": ["uniform_full"],
        },
        {
            "name": "teacher_following_reject",
            "start_frame": int(student_reject["start_frame"]),
            "end_frame": int(student_reject["end_frame"]),
            "variants": ["uniform_full", "first_30", "middle_30", "last_30"],
        },
        {
            "name": "teacher_emit_plus_following",
            "start_frame": int(teacher_emit["start_frame"]),
            "end_frame": int(student_reject["end_frame"]),
            "variants": ["uniform_full", "first_30", "middle_30", "last_30"],
        },
    ]
    anchors = ["torso_center", "mid_shoulder"]

    results: list[dict[str, object]] = []
    for window in windows:
        for anchor in anchors:
            for variant in window["variants"]:
                sampled_indices = _variant_indices(window["start_frame"], window["end_frame"], variant)
                sequence = arrays["feature_vectors"][sampled_indices].astype(np.float32)
                sequence = _rebuild_pose_local_sequence(
                    base_sequence=sequence,
                    arrays=arrays,
                    sampled_indices=sampled_indices,
                    feature_spec=feature_spec,
                    pose_local_anchor=anchor,
                )
                score = _score_sequence(
                    model=model,
                    feature_mode=feature_mode,
                    feature_spec=feature_spec,
                    index_to_label=index_to_label,
                    sequence=sequence,
                )
                results.append(
                    {
                        "window_name": window["name"],
                        "start_frame": int(window["start_frame"]),
                        "end_frame": int(window["end_frame"]),
                        "variant": variant,
                        "pose_local_anchor": anchor,
                        "sampled_frame_indices": sampled_indices.astype(int).tolist(),
                        **score,
                    }
                )

    best_teacher = max(results, key=lambda row: (float(row["teacher_probability"]), float(row["teacher_minus_student_logit_gap"])))
    output_payload = {
        "session_dir": str(session_dir),
        "exact_classification_json": str(exact_json),
        "cache_dir": str(cache_dir),
        "anchors": anchors,
        "results": results,
        "best_teacher_variant": best_teacher,
        "teacher_top1_variant_count": sum(1 for row in results if row["predicted_label"] == "teacher"),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_json}")


if __name__ == "__main__":
    main()
