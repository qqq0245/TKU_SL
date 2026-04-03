from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

REPOS = [
    {
        "name": "Realtime-Sign-Language-Detection-Using-LSTM-Model",
        "category": "01_baseline_lstm",
        "url": "https://github.com/AvishakeAdhikary/Realtime-Sign-Language-Detection-Using-LSTM-Model",
    },
    {
        "name": "Real-Time-Sign-Language-Recognition",
        "category": "02_hand_landmark_or_static_sign",
        "url": "https://github.com/MonzerDev/Real-Time-Sign-Language-Recognition",
    },
    {
        "name": "Indian-Sign-Language-Detection",
        "category": "02_hand_landmark_or_static_sign",
        "url": "https://github.com/MaitreeVaria/Indian-Sign-Language-Detection",
    },
    {
        "name": "Real-Time-Sign-Language",
        "category": "03_detection_yolo",
        "url": "https://github.com/paulinamoskwa/Real-Time-Sign-Language",
    },
    {
        "name": "HA-SLR-GCN",
        "category": "04_skeleton_gcn",
        "url": "https://github.com/snorlaxse/HA-SLR-GCN",
    },
    {
        "name": "SLGTformer",
        "category": "05_transformer_sign",
        "url": "https://github.com/neilsong/SLGTformer",
    },
    {
        "name": "SignSense",
        "category": "06_full_pipeline_or_llm",
        "url": "https://github.com/DEV-D-GR8/SignSense",
    },
]

KEY_FILES = [
    "README.md",
    "readme.md",
    "requirements.txt",
    "environment.yml",
    "setup.py",
    "pyproject.toml",
]

SCRIPT_HINTS = [
    "train",
    "infer",
    "predict",
    "detect",
    "webcam",
    "camera",
    "preprocess",
    "dataset",
    "model",
]


def list_key_files(repo_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in KEY_FILES:
        for path in repo_dir.rglob(pattern):
            found.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return sorted(set(found))


def list_script_candidates(repo_dir: Path) -> list[str]:
    matches: list[str] = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".ipynb", ".sh", ".yaml", ".yml"}:
            continue
        lower_name = path.name.lower()
        if any(hint in lower_name for hint in SCRIPT_HINTS):
            matches.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return sorted(set(matches))


def main() -> None:
    report = []
    for repo in REPOS:
        repo_dir = ROOT / repo["category"] / repo["name"]
        exists = repo_dir.exists()
        report.append(
            {
                **repo,
                "path": str(repo_dir.relative_to(ROOT)).replace("\\", "/"),
                "exists": exists,
                "key_files": list_key_files(repo_dir) if exists else [],
                "script_candidates": list_script_candidates(repo_dir) if exists else [],
            }
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
