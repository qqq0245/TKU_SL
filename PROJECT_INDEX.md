# Project Index

## Datasets
- `datasets/raw/00_videos`
  - 原始單字影片總庫。
- `datasets/training_variants/00_videos_training_30`
  - 30 字基礎訓練子集。
- `datasets/training_variants/00_videos_training_30_*`
  - 各種資料擴增、句子補強、失敗案例補強版本。
- `datasets/sentences/00_sentence_videos_50`
  - 50 部合理句子影片。
- `datasets/sentences/00_sentence_word_segments_*`
  - 從句子影片切出的單字片段資料。
- `datasets/recorded/webcam_30_words`
  - webcam 錄製資料。

## Metadata
- `metadata/asl_training_answers.csv`
  - ASL gloss / 英文 / 中文對照表。
- `metadata/sentence_video_manifest_50.csv`
  - 50 句句子影片 manifest。
- `metadata/sentence_word_segments_manifest_50.csv`
  - 句子切詞後的單字片段 manifest。
- `metadata/training_30_vocabulary.csv`
  - 30 字詞彙表。
- `metadata/webcam_30_words_manifest.csv`
  - webcam 錄製逐檔 manifest。
- `metadata/webcam_30_words_manifest_label_summary.csv`
  - webcam 錄製每個標籤的摘要統計。

## Reports
- `reports/regression`
  - 句子離線回歸結果。
- `reports/inventories`
  - 資料掃描清單，例如 `00_videos_words.txt`。
- `reports/continuous_inference_repair_progress.md`
  - continuous inference 修補進度主紀錄。

## Experiments
- `experiments/training_experiments_log.csv`
  - 訓練實驗結構化紀錄。
- `experiments/training_experiments_log.md`
  - 訓練實驗摘要說明。

## Test Assets
- `test_assets/test_video.mp4`
  - 單一測試影片。
- `test_assets/test_video_segments`
  - 測試影片切段結果。

## Main Workspace
- `integration_workspace`
  - 主要程式、模型、dataset pipeline、artifacts。

## Local Assets
- `local_assets/videos/root_captures`
  - 從根目錄移出的手動錄製影片，保留本地，不納入 Git 版。

## Git Upload Version
- `scripts/export_code_only_snapshot.ps1`
  - 產生乾淨的 code-only GitHub 上傳版，排除影片、模型、`npz`、cache 與大型實驗產物。

## Program Defaults Updated
- `run_sentence_interface.py`
  - 預設改讀 `metadata/asl_training_answers.csv`。
- `run_webcam_data_collection.py`
  - 預設輸出改到 `datasets/recorded/webcam_30_words`，manifest 改到 `metadata`。
- `build_sentence_video_dataset.py`
  - 預設來源改到 `datasets/training_variants/00_videos_training_30`。
- `extract_word_clips_from_sentence_videos.py`
  - 預設句子 manifest 與切詞 manifest 改到 `metadata`。
- `config.py`
  - 預設原始單字來源改到 `datasets/raw/00_videos`。
