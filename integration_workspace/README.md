# Integration Workspace

這個工作區是手語辨識專題的第一版最小可行 baseline。

主流程：

`webcam -> landmarks -> normalization -> sequence -> dataset -> LSTM training -> realtime inference`

## 快速開始

```bash
pip install -r requirements.txt
python run_capture.py --label hello --num-sequences 5 --sequence-length 30
python run_capture.py --label no_sign --num-sequences 20 --sequence-length 30
python train_lstm.py
python run_inference.py
```

## 功能

- webcam 擷取
- MediaPipe Holistic 關鍵點抽取
- 左手 / 右手 / 上半身 pose landmarks 統一輸出
- torso-relative normalization
- 固定長度 sequence builder
- `.npz` dataset export
- PyTorch LSTM baseline
- realtime inference 骨架
- 未來 GCN 匯出接口
- inference control layer
  - no_sign / idle 支援
  - confidence threshold
  - temporal smoothing
  - valid-hand / sequence gate

## 為什麼需要 no_sign / idle 類別

如果沒有 `no_sign`，模型在你沒比手勢時也會硬猜成已知類別，例如 `hello` 或 `thanks`。

建議把 `no_sign` 視為正式類別，並收集至少接近主要類別數量的樣本。

### 建議收集的 no_sign 情境

- 人坐在鏡頭前但不比手語
- 手放下
- 自然姿態
- 只有頭或肩膀有小動作
- 手在畫面內隨機移動，但不是既有手勢

### 建議指令

```bash
python run_capture.py --label no_sign --num-sequences 20 --sequence-length 30
```

## 主要文件

- `docs/architecture.md`
- `docs/dataset_format.md`
- `docs/migration_to_gcn.md`
- `docs/inference_control.md`
- `docs/large_scale_dataset_pipeline.md`
- `docs/pipeline_operations.md`

## 2000 單字影片批次資料管線

目前已新增大型資料集流程：

- `dataset_pipeline/raw_videos/`
- `dataset_pipeline/manifests/`
- `dataset_pipeline/landmarks_cache/`
- `dataset_pipeline/processed_sequences/`
- `dataset_pipeline/splits/`
- `dataset_pipeline/logs/`

主要腳本：

- `scripts/build_label_map.py`
- `scripts/scan_video_dataset.py`
- `scripts/extract_landmarks_batch.py`
- `scripts/export_sequences_batch.py`
- `scripts/build_splits.py`
- `scripts/build_subset_experiment.py`
- `scripts/build_class_coverage_report.py`
- `scripts/build_failure_report.py`
- `scripts/pipeline_status.py`

建議先跑：

```bash
python scripts/build_label_map.py
python scripts/scan_video_dataset.py
python scripts/extract_landmarks_batch.py --limit 12 --skip-existing
python scripts/export_sequences_batch.py --limit 12 --skip-existing
python scripts/build_splits.py
python scripts/build_subset_experiment.py --top-k 20 --output-name top20_debug
python train_multibranch.py --data-dir dataset_pipeline/processed_sequences --train-split-csv dataset_pipeline/splits/subsets/top20_debug/train.csv --val-split-csv dataset_pipeline/splits/subsets/top20_debug/val.csv
```
