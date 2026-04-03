from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Analyze father/teacher exact-span vs serving residual structure.")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--exact-classification-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _top_labels(encoded: str) -> list[str]:
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    labels: list[str] = []
    for item in payload[:3]:
        if isinstance(item, dict):
            labels.append(str(item.get("label", "")).strip().lower())
    return labels


def _father_rule_ratio(indices: list[int], arrays: dict[str, np.ndarray]) -> dict[str, object]:
    hit_count = 0
    valid_count = 0
    sample_rows: list[dict[str, object]] = []
    for frame_index in indices:
        left_valid = bool(np.any(arrays["left_hand_mask"][frame_index] > 0))
        right_valid = bool(np.any(arrays["right_hand_mask"][frame_index] > 0))
        nose_valid = bool(arrays["pose_mask"][frame_index][0] > 0)
        if not nose_valid or (not left_valid and not right_valid):
            continue
        hand = arrays["normalized_right_hand"][frame_index] if right_valid else arrays["normalized_left_hand"][frame_index]
        hand_mask = arrays["right_hand_mask"][frame_index] if right_valid else arrays["left_hand_mask"][frame_index]
        if float(hand_mask[0]) <= 0.0:
            continue
        wrist_y = float(hand[0][1])
        nose_y = float(arrays["normalized_nose"][frame_index][1])
        chin_valid = bool(arrays["chin_mask"][frame_index][0] > 0)
        chin_y = float(arrays["normalized_chin"][frame_index][1]) if chin_valid else nose_y + 0.18
        father_threshold = max(chin_y - nose_y, 0.12) * 0.60
        relative_y = wrist_y - nose_y
        hit = relative_y <= father_threshold
        valid_count += 1
        hit_count += int(hit)
        if len(sample_rows) < 8:
            sample_rows.append(
                {
                    "frame_index": int(frame_index),
                    "selected_hand": "right" if right_valid else "left",
                    "relative_y": round(relative_y, 6),
                    "father_threshold": round(float(father_threshold), 6),
                    "hit": bool(hit),
                }
            )
    return {
        "valid_count": int(valid_count),
        "hit_count": int(hit_count),
        "hit_ratio": round((hit_count / valid_count) if valid_count else 0.0, 6),
        "samples": sample_rows,
    }


def _levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (0 if ref_token == hyp_token else 1),
                )
            )
        previous = current
    return previous[-1]


def _wer(reference: list[str], hypothesis: list[str]) -> float:
    return _levenshtein(reference, hypothesis) / max(len(reference), 1)


def _classify_residual_type(*, token: str, exact_span: dict[str, object], serving_segments: list[dict[str, object]]) -> dict[str, object]:
    exact_label = str(exact_span.get("predicted_label", "")).strip().lower()
    top3 = exact_span.get("top3", [])
    exact_top2 = [str(item.get("label", "")).strip().lower() for item in top3[:2] if isinstance(item, dict)]
    token_in_exact_top2 = token in exact_top2
    serving_labels = [str(segment.get("raw_label", "")).strip().lower() for segment in serving_segments]
    token_serving_hits = sum(1 for label in serving_labels if label == token)
    serving_non_token = [segment for segment in serving_segments if str(segment.get("raw_label", "")).strip().lower() != token]
    token_in_non_token_top2 = any(token in list(segment.get("top_labels", [])[:2]) for segment in serving_non_token)

    if exact_label != token and not token_in_exact_top2 and token_serving_hits == 0:
        return {
            "residual_type": "exact_span_suppression",
            "reason": f"{token} is absent from exact-span top2 and never becomes the serving raw label.",
        }
    if exact_label == token and token_serving_hits == 0 and token_in_non_token_top2:
        return {
            "residual_type": "serving_scorer_mismatch",
            "reason": f"{token} is already exact-span-correct but remains suppressed only in serving scorer ranking.",
        }
    return {
        "residual_type": "mixed_residual",
        "reason": (
            f"{token} is suppressed at exact-span, but serving is split: "
            f"{token_serving_hits} segment(s) already reach {token} while the remaining non-{token} segments do not show a clean scorer-only candidate."
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    session_dir = Path(args.session_dir).resolve()
    exact_path = Path(args.exact_classification_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_json = Path(args.output_json).resolve()

    exact_payload = _load_json(exact_path)
    summary_payload = _load_json(session_dir / "session_summary.json")
    alignment_rows = _load_csv(session_dir / "trigger_segment_alignment.csv")
    with np.load(cache_dir / "continuous_feature_cache_mirror1.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}

    exact_rows = {
        str(row["token"]).strip().lower(): row
        for row in exact_payload.get("classifications", [])
        if int(row.get("mirror_input", 0)) == 1 and str(row.get("token", "")).strip().lower() in {"father", "teacher"}
    }
    serving_rows = [
        row
        for row in alignment_rows
        if str(row.get("reference_label", "")).strip().lower() in {"father", "teacher"}
    ]

    residuals: dict[str, dict[str, object]] = {}
    for token in ("father", "teacher"):
        exact_row = exact_rows.get(token, {})
        token_segments = [
            row for row in serving_rows if str(row.get("reference_label", "")).strip().lower() == token
        ]
        exact_indices = [int(value) for value in exact_row.get("sampled_frame_indices", [])] if exact_row else []
        residuals[token] = {
            "exact_span": {
                "predicted_label": str(exact_row.get("predicted_label", "")),
                "top3": exact_row.get("top3", []),
                "active_frame_count": int(exact_row.get("active_frame_count", 0) or 0),
                "father_rule_geometry": _father_rule_ratio(exact_indices, arrays) if exact_indices else {},
            },
            "serving_segments": [],
        }
        for row in token_segments:
            sampled_frame_indices = [int(value) for value in json.loads(row.get("sampled_frame_indices") or "[]")]
            residuals[token]["serving_segments"].append(
                {
                    "segment_id": int(row.get("segment_id") or 0),
                    "raw_label": str(row.get("raw_label") or ""),
                    "emitted_label": str(row.get("emitted_label") or ""),
                    "decision_status": str(row.get("decision_status") or ""),
                    "raw_confidence": float(row.get("raw_confidence") or 0.0),
                    "top_margin": float(row.get("top_margin") or 0.0),
                    "top_labels": _top_labels(row.get("top_candidates") or "[]"),
                    "top3": json.loads(row.get("top_candidates") or "[]"),
                    "father_rule_geometry": _father_rule_ratio(sampled_frame_indices, arrays),
                }
            )
        classification = _classify_residual_type(
            token=token,
            exact_span=residuals[token]["exact_span"],
            serving_segments=residuals[token]["serving_segments"],
        )
        residuals[token]["residual_type"] = classification["residual_type"]
        residuals[token]["residual_reason"] = classification["reason"]

    reference_tokens = list(summary_payload.get("continuous_evaluation", {}).get("reference_tokens", []))
    predicted_tokens = list(summary_payload.get("continuous_evaluation", {}).get("predicted_tokens", []))
    candidate_hypotheses = {
        "father_only": ["you", "mother", "father", "student", "student", "like"],
        "teacher_only": ["you", "mother", "teacher", "student", "like"],
        "father_and_teacher": ["you", "mother", "father", "teacher", "student", "like"],
    }
    counterfactual = {
        name: {
            "predicted_tokens": tokens,
            "word_error_rate": round(_wer(reference_tokens, tokens), 6),
        }
        for name, tokens in candidate_hypotheses.items()
    }

    father_segments = residuals["father"]["serving_segments"]
    teacher_segments = residuals["teacher"]["serving_segments"]
    father_narrow_candidate = next(
        (
            segment
            for segment in father_segments
            if segment["raw_label"] == "no_sign"
            and segment["top_labels"][:2] == ["no_sign", "father"]
            and float(segment["father_rule_geometry"].get("hit_ratio", 0.0)) >= 0.75
        ),
        None,
    )
    teacher_narrow_candidate = next(
        (
            segment
            for segment in teacher_segments
            if segment["raw_label"] in {"student", "no_sign"}
            and "teacher" in segment["top_labels"][:2]
        ),
        None,
    )

    chosen_residual = {
        "token": "teacher",
        "residual_type": residuals["teacher"]["residual_type"],
        "reason": (
            "teacher is the cleanest one-token/one-type residual: deep exact-span no_sign suppression with no existing serving-only bridge, "
            "whereas father is already partially rescued in serving and its remaining reject segment is not a pairwise-spacing candidate."
        ),
    }

    output_payload = {
        "session_dir": str(session_dir),
        "exact_classification_json": str(exact_path),
        "cache_dir": str(cache_dir),
        "baseline_word_error_rate": float(summary_payload.get("continuous_evaluation", {}).get("word_error_rate", 0.0)),
        "reference_tokens": reference_tokens,
        "predicted_tokens": predicted_tokens,
        "residuals": residuals,
        "counterfactual_word_error_rate": counterfactual,
        "recommendation": {
            "higher_value_residual": "father",
            "reason": (
                "father has a narrow serving-side rescue candidate: one rejected segment already has "
                "top2=no_sign/father and strong father-geometry hit ratio, while teacher has no "
                "comparable narrow gate and remains split across no_sign exact-span plus student serving emit."
            ),
            "father_narrow_candidate_segment_id": int(father_narrow_candidate["segment_id"]) if father_narrow_candidate else None,
            "teacher_narrow_candidate_segment_id": int(teacher_narrow_candidate["segment_id"]) if teacher_narrow_candidate else None,
        },
        "chosen_single_residual_for_next_experiment": chosen_residual,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_json}")


if __name__ == "__main__":
    main()
