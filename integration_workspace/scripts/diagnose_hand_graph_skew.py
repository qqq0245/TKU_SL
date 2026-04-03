from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.dataset.continuous_feature_cache import load_continuous_feature_cache
from src.models.feature_slices import split_feature_tensor
from src.models.inference_utils_multibranch import load_multibranch_checkpoint


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Diagnose hand-local landmark graph sensitivity on cached exact spans.")
    parser.add_argument("--spans-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "artifacts_webcam9_relative_coord_v1" / "models" / "multibranch_baseline.pt"),
    )
    parser.add_argument("--mirror-input", type=int, default=1)
    return parser


def _hand_graph_indices() -> list[int]:
    channels = int(CONFIG.channels)
    left_xyz = int(CONFIG.left_hand_nodes) * channels
    right_xyz = int(CONFIG.right_hand_nodes) * channels
    pose_nodes = len(CONFIG.pose_indices)
    pose_xyz = pose_nodes * channels
    left_mask_start = left_xyz + right_xyz + pose_xyz
    right_mask_start = left_mask_start + int(CONFIG.left_hand_nodes)
    indices = list(range(0, left_xyz + right_xyz))
    indices.extend(range(left_mask_start, left_mask_start + int(CONFIG.left_hand_nodes)))
    indices.extend(range(right_mask_start, right_mask_start + int(CONFIG.right_hand_nodes)))
    return indices


@torch.no_grad()
def _predict(model, sequence: np.ndarray, feature_mode: str, feature_spec: dict, index_to_label: dict[int, str]) -> dict[str, object]:
    tensor = torch.from_numpy(sequence).unsqueeze(0)
    parts = split_feature_tensor(tensor, feature_mode, feature_spec=feature_spec)
    logits = model(parts["skeleton_stream"], parts["location_stream"], parts["motion_stream"])
    logits_np = logits.squeeze(0).detach().cpu().numpy()
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    top_indices = np.argsort(-probabilities)[:3]
    top3 = [
        {
            "label": index_to_label[int(index)],
            "confidence": round(float(probabilities[int(index)]), 6),
            "logit": round(float(logits_np[int(index)]), 6),
        }
        for index in top_indices
    ]
    return {
        "predicted_label": top3[0]["label"] if top3 else "",
        "top3": top3,
    }


def main() -> None:
    args = build_parser().parse_args()
    spans_path = Path(args.spans_json).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_path = Path(args.output_json).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    mirror_input = bool(args.mirror_input)

    payload = json.loads(spans_path.read_text(encoding="utf-8"))
    spans = payload.get("token_span_hypothesis", [])
    cache_path = cache_dir / f"continuous_feature_cache_mirror{int(mirror_input)}.npz"
    metadata, arrays = load_continuous_feature_cache(cache_path)

    device = torch.device("cpu")
    model, checkpoint = load_multibranch_checkpoint(checkpoint_path, device)
    if model is None or checkpoint is None:
        raise RuntimeError(f"Unable to load checkpoint: {checkpoint_path}")
    model.eval()

    feature_mode = str(checkpoint["feature_mode"])
    feature_spec = checkpoint.get("feature_spec")
    index_to_label = checkpoint["index_to_label"]
    feature_vectors = arrays["feature_vectors"]
    hand_indices = _hand_graph_indices()

    rows: list[dict[str, object]] = []
    changed_counter: Counter[str] = Counter()
    for span in spans:
        start_frame = int(span["start_frame"])
        end_frame = min(int(span["end_frame"]), int(metadata["readable_frame_count"]) - 1)
        sampled_indices = (
            np.linspace(start_frame, end_frame, 30).round().astype(int)
            if end_frame > start_frame
            else np.full((30,), start_frame, dtype=int)
        )
        full_sequence = feature_vectors[sampled_indices].astype(np.float32)
        ablated_sequence = full_sequence.copy()
        ablated_sequence[:, hand_indices] = 0.0

        full_pred = _predict(model, full_sequence, feature_mode, feature_spec, index_to_label)
        ablated_pred = _predict(model, ablated_sequence, feature_mode, feature_spec, index_to_label)
        changed = int(full_pred["predicted_label"] != ablated_pred["predicted_label"])
        if changed:
            changed_counter[str(span["token"])] += 1
        rows.append(
            {
                "token": span["token"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "full_predicted_label": full_pred["predicted_label"],
                "ablated_predicted_label": ablated_pred["predicted_label"],
                "changed_after_hand_graph_ablation": changed,
                "full_top3": full_pred["top3"],
                "ablated_top3": ablated_pred["top3"],
            }
        )

    summary = {
        "mirror_input": int(mirror_input),
        "row_count": len(rows),
        "changed_after_hand_graph_ablation_count": int(sum(row["changed_after_hand_graph_ablation"] for row in rows)),
        "changed_tokens": dict(changed_counter),
        "hand_graph_feature_columns": hand_indices,
    }
    output_payload = {
        "spans_json": str(spans_path),
        "cache_path": str(cache_path),
        "checkpoint_path": str(checkpoint_path),
        "summary": summary,
        "rows": rows,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {output_path}")


if __name__ == "__main__":
    main()
