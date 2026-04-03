from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_you_like_serving_margin import (  # noqa: E402
    _configure_engine,
    _load_json,
    _rebuild_pose_local_sequence,
    _safe_float,
    _score_stage,
)
from src.dataset.continuous_feature_cache import load_continuous_feature_cache  # noqa: E402


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Test whether narrow exact-span pose-context reconstruction can recover want.")
    parser.add_argument("--exact-classification-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prototype-stats-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gesture-profile", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--single-group", choices=("pose_coords_only", "pose_context_graph"))
    parser.add_argument("--single-alpha", type=float)
    parser.add_argument("--single-vector-replacement", action="store_true")
    return parser


def _group_indices() -> dict[str, np.ndarray]:
    channels = 3
    left_xyz = 21 * channels
    right_xyz = 21 * channels
    pose_xyz = 9 * channels
    left_mask_start = left_xyz + right_xyz + pose_xyz
    right_mask_start = left_mask_start + 21
    pose_mask_start = right_mask_start + 21
    pose_coords = np.arange(left_xyz + right_xyz, left_xyz + right_xyz + pose_xyz, dtype=np.int32)
    pose_context = np.concatenate(
        [
            pose_coords,
            np.arange(pose_mask_start, pose_mask_start + 9, dtype=np.int32),
        ]
    )
    return {
        "pose_coords_only": pose_coords,
        "pose_context_graph": pose_context,
    }


def _load_sequences(paths: list[str]) -> list[np.ndarray]:
    sequences: list[np.ndarray] = []
    for raw_path in paths:
        sample_path = Path(raw_path)
        if not sample_path.exists():
            continue
        with np.load(sample_path, allow_pickle=False) as payload:
            sequence = payload["sequence"].astype(np.float32)
        if sequence.shape[0] != 30:
            sample_positions = np.linspace(0, sequence.shape[0] - 1, 30).round().astype(np.int32)
            sequence = sequence[sample_positions]
        sequences.append(sequence.astype(np.float32, copy=False))
    return sequences


def _want_stage_metrics(stage: dict[str, object]) -> dict[str, object]:
    top_logits = {str(item["label"]).strip().lower(): float(item["logit"]) for item in stage.get("top_logits", [])}
    top_candidates = {str(item["label"]).strip().lower(): float(item["confidence"]) for item in stage.get("top_candidates", [])}
    return {
        "raw_label": str(stage.get("raw_label", "")),
        "raw_confidence": _safe_float(float(stage.get("raw_confidence", 0.0))),
        "top_margin": _safe_float(float(stage.get("top_margin", 0.0))),
        "want_probability": _safe_float(top_candidates.get("want", 0.0)),
        "student_probability": _safe_float(top_candidates.get("student", 0.0)),
        "nosign_probability": _safe_float(top_candidates.get("no_sign", 0.0)),
        "want_logit": _safe_float(top_logits.get("want", float("-inf"))),
        "student_logit": _safe_float(top_logits.get("student", float("-inf"))),
        "nosign_logit": _safe_float(top_logits.get("no_sign", float("-inf"))),
        "want_minus_student_logit_gap": _safe_float(top_logits.get("want", float("-inf")) - top_logits.get("student", float("-inf"))),
        "want_minus_nosign_logit_gap": _safe_float(top_logits.get("want", float("-inf")) - top_logits.get("no_sign", float("-inf"))),
        "top_candidates": stage.get("top_candidates", []),
        "top_logits": stage.get("top_logits", []),
    }


def _score_sequence(engine, sequence: np.ndarray) -> dict[str, object]:
    return _want_stage_metrics(_score_stage(engine=engine, sequence=sequence))


def _override_group(
    *,
    base_sequence: np.ndarray,
    override_sequence: np.ndarray,
    indices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    rebuilt = base_sequence.astype(np.float32, copy=True)
    rebuilt[:, indices] = ((1.0 - alpha) * rebuilt[:, indices]) + (alpha * override_sequence[:, indices])
    return rebuilt


def _override_group_with_vector(
    *,
    base_sequence: np.ndarray,
    override_vector: np.ndarray,
    indices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    rebuilt = base_sequence.astype(np.float32, copy=True)
    blended = ((1.0 - alpha) * rebuilt[:, indices]) + (alpha * override_vector[np.newaxis, :])
    rebuilt[:, indices] = blended.astype(np.float32, copy=False)
    return rebuilt


def _result_rank_key(item: tuple[str, dict[str, object]]) -> tuple[float, float, float]:
    metrics = item[1]
    return (
        float(metrics.get("want_probability", 0.0)),
        float(metrics.get("want_minus_student_logit_gap", float("-inf"))),
        float(metrics.get("want_minus_nosign_logit_gap", float("-inf"))),
    )


def main() -> None:
    args = build_parser().parse_args()
    exact_classification_path = Path(args.exact_classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    prototype_stats_path = Path(args.prototype_stats_json).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    gesture_profile_path = Path(args.gesture_profile).resolve()
    output_path = Path(args.output_json).resolve()

    exact_payload = _load_json(exact_classification_path)
    stats_payload = _load_json(prototype_stats_path)
    want_row = next(
        row
        for row in exact_payload["classifications"]
        if int(row.get("mirror_input", -1)) == 1 and str(row.get("token", "")).strip().lower() == "want"
    )
    sampled_frame_indices = list(want_row["sampled_frame_indices"])

    cache_path = cache_dir / "continuous_feature_cache_mirror1.npz"
    _metadata, arrays = load_continuous_feature_cache(cache_path)
    feature_vectors = arrays["feature_vectors"].astype(np.float32)

    engine = _configure_engine(
        checkpoint_path=checkpoint_path,
        gesture_profile_path=gesture_profile_path,
        enable_mother_fix=True,
        enable_you_i_fix=True,
    )
    engine.pose_local_anchor = "torso_center"
    engine.enable_you_like_pairwise_calibration = True
    engine.you_like_pairwise_delta = 0.98
    engine.enable_like_i_pairwise_calibration = True
    engine.like_i_pairwise_delta = 2.78

    mid_sequence = feature_vectors[np.asarray(sampled_frame_indices, dtype=np.int32)].astype(np.float32, copy=True)
    torso_sequence = _rebuild_pose_local_sequence(
        feature_vectors=feature_vectors,
        normalized_pose=arrays["normalized_pose"],
        sampled_frame_indices=sampled_frame_indices,
    )

    prototype_summary = stats_payload["prototype_summary"]
    want_sequences = _load_sequences(list(prototype_summary["want"]["sample_paths"]))
    student_sequences = _load_sequences(list(prototype_summary["student"]["sample_paths"]))
    if not want_sequences:
        raise RuntimeError("No want prototype sequences were available for reconstruction testing.")

    want_mean_sequence = np.stack(want_sequences, axis=0).mean(axis=0).astype(np.float32)
    student_mean_sequence = np.stack(student_sequences, axis=0).mean(axis=0).astype(np.float32) if student_sequences else None
    groups = _group_indices()
    single_group = args.single_group
    single_alpha = float(args.single_alpha) if args.single_alpha is not None else None
    if args.single_vector_replacement:
        if single_group is None:
            raise RuntimeError("--single-group is required when --single-vector-replacement is enabled.")
        if single_alpha is None:
            raise RuntimeError("--single-alpha is required when --single-vector-replacement is enabled.")
        if not 0.0 <= single_alpha <= 1.0:
            raise RuntimeError("--single-alpha must be between 0.0 and 1.0.")

    results: dict[str, dict[str, object]] = {
        "torso_center_base": _score_sequence(engine, torso_sequence),
        "mid_shoulder_base": _score_sequence(engine, mid_sequence),
    }

    torso_with_mid_pose = _override_group(
        base_sequence=torso_sequence,
        override_sequence=mid_sequence,
        indices=groups["pose_coords_only"],
        alpha=1.0,
    )
    results["torso_with_mid_pose_coords_only"] = _score_sequence(engine, torso_with_mid_pose)

    if args.single_vector_replacement:
        indices = groups[single_group]
        prototype_vector = want_mean_sequence[:, indices].mean(axis=0).astype(np.float32)
        alpha_key = str(int(round(single_alpha * 100.0))).rjust(3, "0")
        key = f"torso_with_want_single_vector_{single_group}_a{alpha_key}"
        results[key] = _score_sequence(
            engine,
            _override_group_with_vector(
                base_sequence=torso_sequence,
                override_vector=prototype_vector,
                indices=indices,
                alpha=single_alpha,
            ),
        )
        alphas = (single_alpha,)
    else:
        alphas = (0.25, 0.5, 0.75, 1.0)
        for alpha in alphas:
            alpha_key = str(int(alpha * 100)).rjust(3, "0")
            for group_name, indices in groups.items():
                want_key = f"torso_with_want_mean_{group_name}_a{alpha_key}"
                results[want_key] = _score_sequence(
                    engine,
                    _override_group(
                        base_sequence=torso_sequence,
                        override_sequence=want_mean_sequence,
                        indices=indices,
                        alpha=alpha,
                    ),
                )
                if student_mean_sequence is not None and alpha == 1.0:
                    student_key = f"torso_with_student_mean_{group_name}_a{alpha_key}"
                    results[student_key] = _score_sequence(
                        engine,
                        _override_group(
                            base_sequence=torso_sequence,
                            override_sequence=student_mean_sequence,
                            indices=indices,
                            alpha=alpha,
                        ),
                    )

        for sample_index, want_sequence in enumerate(want_sequences, start=1):
            sample_name = f"want_sample_{sample_index:02d}"
            for group_name, indices in groups.items():
                key = f"torso_with_{sample_name}_{group_name}_a100"
                results[key] = _score_sequence(
                    engine,
                    _override_group(
                        base_sequence=torso_sequence,
                        override_sequence=want_sequence,
                        indices=indices,
                        alpha=1.0,
                    ),
                )

    best_override_key, best_override_metrics = max(
        (
            item
            for item in results.items()
            if item[0] not in {"torso_center_base", "mid_shoulder_base"}
        ),
        key=_result_rank_key,
    )
    want_recovered = str(best_override_metrics.get("raw_label", "")).strip().lower() == "want"

    output_payload = {
        "exact_classification_json": str(exact_classification_path),
        "cache_path": str(cache_path),
        "prototype_stats_json": str(prototype_stats_path),
        "sampled_frame_indices": sampled_frame_indices,
        "tested_groups": list(groups.keys()),
        "tested_alphas": list(alphas),
        "want_prototype_count": len(want_sequences),
        "student_prototype_count": len(student_sequences),
        "single_vector_replacement": bool(args.single_vector_replacement),
        "single_group": single_group,
        "single_alpha": single_alpha,
        "results": results,
        "best_override": {
            "candidate_key": best_override_key,
            "metrics": best_override_metrics,
        },
        "want_recovered": want_recovered,
        "conclusion": (
            "No narrow exact-span reconstruction remains viable on the current evidence."
            if not want_recovered
            else "A narrow exact-span pose-context reconstruction candidate exists and should be considered before downgrading want."
        ),
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
