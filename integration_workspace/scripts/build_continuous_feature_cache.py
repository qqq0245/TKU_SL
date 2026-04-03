from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.continuous_feature_cache import build_continuous_feature_cache, save_continuous_feature_cache
from src.models.inference_utils_multibranch import load_multibranch_checkpoint


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build reusable full-video continuous landmark / feature caches.")
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "artifacts_webcam9_relative_coord_v1" / "models" / "multibranch_baseline.pt"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mirror-modes",
        default="1",
        help="Comma-separated mirror_input values to build, e.g. 1 or 0,1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_mirror_modes(raw_value: str) -> list[bool]:
    modes: list[bool] = []
    seen: set[int] = set()
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value not in {0, 1}:
            raise ValueError(f"Unsupported mirror mode: {token}")
        if value in seen:
            continue
        seen.add(value)
        modes.append(bool(value))
    if not modes:
        raise ValueError("At least one mirror mode is required")
    return modes


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.video).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, checkpoint = load_multibranch_checkpoint(checkpoint_path, device)
    if model is None or checkpoint is None:
        raise RuntimeError(f"Unable to load checkpoint: {checkpoint_path}")

    feature_mode = str(checkpoint["feature_mode"])
    feature_spec = checkpoint.get("feature_spec")
    mirror_modes = _parse_mirror_modes(args.mirror_modes)

    cache_records: list[dict[str, object]] = []
    for mirror_input in mirror_modes:
        cache_path = output_dir / f"continuous_feature_cache_mirror{int(mirror_input)}.npz"
        metadata_path = cache_path.with_suffix(".json")
        if cache_path.exists() and metadata_path.exists() and not args.overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cache_records.append(
                {
                    "mirror_input": int(mirror_input),
                    "cache_path": str(cache_path),
                    "metadata_path": str(metadata_path),
                    "frame_count": int(metadata.get("readable_frame_count", 0)),
                    "status": "reused",
                }
            )
            print(f"[SKIP] Reusing existing cache {cache_path}")
            continue

        print(f"[BUILD] mirror_input={int(mirror_input)} -> {cache_path}")
        cache_payload = build_continuous_feature_cache(
            video_path,
            mirror_input=mirror_input,
            feature_mode=feature_mode,
            feature_spec=feature_spec,
        )
        save_continuous_feature_cache(cache_payload, cache_path)
        metadata = cache_payload["metadata"]
        cache_records.append(
            {
                "mirror_input": int(mirror_input),
                "cache_path": str(cache_path),
                "metadata_path": str(metadata_path),
                "frame_count": int(metadata["readable_frame_count"]),
                "status": "built",
            }
        )

    manifest = {
        "video_path": str(video_path),
        "checkpoint_path": str(checkpoint_path),
        "feature_mode": feature_mode,
        "feature_spec": feature_spec,
        "mirror_modes": [int(value) for value in mirror_modes],
        "cache_records": cache_records,
    }
    manifest_path = output_dir / "continuous_feature_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {manifest_path}")


if __name__ == "__main__":
    main()
