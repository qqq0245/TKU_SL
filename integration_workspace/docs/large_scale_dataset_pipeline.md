# Large Scale Dataset Pipeline

## 1. vocabulary list 結構

目前詞表來源：

- `00_videos_vocabulary_list.csv`
- `00_videos_vocabulary_list.md`

主要欄位：

- `index`
- `english_label`
- `zh_tw_translation`
- `video_count`

## 2. raw video 準備方式

建議資料夾結構：

```text
dataset_pipeline/raw_videos/
├─ word_a/
│  ├─ 00001.mp4
│  ├─ 00002.mp4
├─ word_b/
│  ├─ 00001.mp4
```

這一版另外支援 adapter 模式：

- 如果 `dataset_pipeline/raw_videos/` 是空的
- script 會自動改掃描 `sign_language_research/00_videos/`

也就是你目前現有的 2000 單字影片不需要先搬檔。

## 3. label map 建立方式

```bash
python scripts/build_label_map.py
```

輸出：

- `dataset_pipeline/manifests/label_map.json`
- `dataset_pipeline/manifests/vocabulary_manifest.csv`

欄位至少包含：

- `class_id`
- `english_label`
- `label_slug`
- `zh_tw_translation`
- `video_count`

## 4. landmarks batch extraction 流程

```bash
python scripts/extract_landmarks_batch.py --skip-existing
```

流程：

1. 讀 `video_manifest.csv`
2. 逐支影片用 MediaPipe Holistic 抽 landmarks
3. 同時保存 raw landmarks 與 normalized landmarks
4. 輸出到 `dataset_pipeline/landmarks_cache/`

cache 內容至少包含：

- `raw_left_hand`
- `raw_right_hand`
- `raw_pose`
- `raw_mouth_center`
- `raw_chin`
- `left_hand_mask`
- `right_hand_mask`
- `pose_mask`
- `mouth_mask`
- `chin_mask`
- `normalized_left_hand`
- `normalized_right_hand`
- `normalized_pose`
- `frame_valid_mask`

## 5. sequence export 流程

```bash
python scripts/export_sequences_batch.py --sequence-length 30 --stride 30 --skip-existing
```

流程：

1. 從 landmarks cache 重新建 feature
2. 使用正式 feature mode：
   - `landmarks_plus_location_motion`
3. 切成固定長度 sequence
4. 輸出 `.npz` 到 `dataset_pipeline/processed_sequences/`

每筆 `.npz` 至少包含：

- `sequence`
- `class_label`
- `class_id`
- `sample_id`
- `metadata_json`
- `frame_valid_mask`

## 6. split 建立方式

```bash
python scripts/build_splits.py
```

輸出：

- `dataset_pipeline/splits/train.csv`
- `dataset_pipeline/splits/val.csv`
- `dataset_pipeline/splits/test.csv`

欄位：

- `sample_id`
- `sample_path`
- `class_id`
- `english_label`
- `zh_tw_translation`
- `feature_mode`
- `feature_dim`
- `sequence_length`
- `source_video_path`

## 7. 建議先做 20 / 50 類驗證

不要一開始就直接吃完整 1999 類。

建議順序：

1. 先抽 landmarks cache
2. 先 export 一小批 sequence
3. 用 `build_subset_experiment.py` 做 top-k 類別子集
4. 先跑 20 類或 50 類驗證訓練

例如：

```bash
python scripts/build_subset_experiment.py --top-k 20 --output-name top20_debug
python train_multibranch.py --data-dir dataset_pipeline/processed_sequences --train-split-csv dataset_pipeline/splits/subsets/top20_debug/train.csv --val-split-csv dataset_pipeline/splits/subsets/top20_debug/val.csv
```

## 8. 之後如何擴到完整 1999 類

建議流程：

1. 先建立完整 `label_map`
2. 生成完整 `video_manifest`
3. 分批跑 `extract_landmarks_batch.py`
4. 分批跑 `export_sequences_batch.py`
5. 建完整 `train/val/test` split
6. 先做 20 類、50 類、100 類
7. 再開始 1999 類訓練

## 9. 哪些步驟最容易出錯

- label 名稱和資料夾名稱不一致
- 影片損毀或無法被 OpenCV 讀取
- landmarks cache 尚未完成就先 export sequence
- feature mode 與訓練 checkpoint 不一致
- subset split 用的是舊資料或錯的 `data_dir`

## 10. 已驗證的小規模流程

我已經用小規模影片跑通以下流程：

1. `build_label_map.py`
2. `scan_video_dataset.py`
3. `extract_landmarks_batch.py --limit 12`
4. `export_sequences_batch.py --limit 12`
5. `build_splits.py`
6. `build_subset_experiment.py --top-k 3`
7. `train_multibranch.py` 讀 subset split 訓練成功

## 11. 第 5.6 階段補強

目前另外補上：

- `pipeline_state.json`
- `landmarks_failed.csv`
- `sequences_failed.csv`
- `class_coverage_report.csv`
- `failure_report.csv`
- `pipeline_status.py`

也就是這套管線現在支援：

- resume / skip existing
- retry failed
- progress state
- coverage report
- failure report
- subset 依 usable sequences 篩選
