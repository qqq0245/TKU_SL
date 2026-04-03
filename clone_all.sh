#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clone_repo() {
  local url="$1"
  local target_dir="$2"

  mkdir -p "$(dirname "$target_dir")"
  if [ -d "$target_dir/.git" ]; then
    echo "[skip] $target_dir already exists"
  else
    echo "[clone] $url -> $target_dir"
    git clone "$url" "$target_dir"
  fi
}

clone_repo \
  "https://github.com/AvishakeAdhikary/Realtime-Sign-Language-Detection-Using-LSTM-Model" \
  "$ROOT_DIR/01_baseline_lstm/Realtime-Sign-Language-Detection-Using-LSTM-Model"

clone_repo \
  "https://github.com/MonzerDev/Real-Time-Sign-Language-Recognition" \
  "$ROOT_DIR/02_hand_landmark_or_static_sign/Real-Time-Sign-Language-Recognition"

clone_repo \
  "https://github.com/MaitreeVaria/Indian-Sign-Language-Detection" \
  "$ROOT_DIR/02_hand_landmark_or_static_sign/Indian-Sign-Language-Detection"

clone_repo \
  "https://github.com/paulinamoskwa/Real-Time-Sign-Language" \
  "$ROOT_DIR/03_detection_yolo/Real-Time-Sign-Language"

clone_repo \
  "https://github.com/snorlaxse/HA-SLR-GCN" \
  "$ROOT_DIR/04_skeleton_gcn/HA-SLR-GCN"

clone_repo \
  "https://github.com/neilsong/SLGTformer" \
  "$ROOT_DIR/05_transformer_sign/SLGTformer"

clone_repo \
  "https://github.com/DEV-D-GR8/SignSense" \
  "$ROOT_DIR/06_full_pipeline_or_llm/SignSense"
