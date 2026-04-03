# Architecture

## 目標

目前工作區已進入第五階段，整體架構從單一路徑 baseline 發展為：

- landmarks / location / motion 三段 feature
- multi-branch fusion classifier
- skeleton branch 可切換：
  - `LSTM skeleton branch`
  - `GCN skeleton branch`
  - `ST-GCN skeleton branch`
- inference control layer

主流程：

1. `run_capture.py`
2. `src/camera/webcam_stream.py`
3. `src/landmarks/holistic_extractor.py`
4. `src/landmarks/normalization.py`
5. `src/landmarks/location_features.py`
6. `src/landmarks/motion_features.py`
7. `src/landmarks/feature_builder.py`
8. `src/dataset/sequence_builder.py`
9. `src/dataset/dataset_writer.py`
10. `train_lstm.py`
11. `train_multibranch.py`
12. `src/models/multibranch_model.py`
13. `src/models/gcn_skeleton_branch.py`
14. `src/pipeline/training_pipeline_multibranch.py`
15. `run_inference_multibranch.py`
16. `src/pipeline/inference_postprocess.py`

大型資料集主流程：

1. `scripts/build_label_map.py`
2. `scripts/scan_video_dataset.py`
3. `scripts/extract_landmarks_batch.py`
4. `scripts/export_sequences_batch.py`
5. `scripts/build_splits.py`
6. `scripts/build_subset_experiment.py`
7. `train_multibranch.py --train-split-csv ...`

## 模組

- `camera`: webcam I/O
- `landmarks`: holistic 抽點、正規化、location/motion 特徵建立
- `dataset`: sequence 緩衝、sample 寫入、PyTorch 讀取
- `dataset_pipeline`: 大型影片資料工程輸出
- `models`: 單一路徑 LSTM、multi-branch、GCN skeleton branch
- `pipeline`: capture、training、inference control、GCN export
- `utils`: logger、paths、labels

## Feature Flow

1. raw frame
2. MediaPipe Holistic extraction
3. frame record
4. normalization
5. landmarks stream
6. location stream
7. motion stream
8. sequence builder
9. model encoder
10. inference control layer

## 模型路線

### 舊版單一路徑 LSTM

- input: `(B, T, 291)`
- encoder: 單一 LSTM
- classifier: 單一 head

### 第三階段 multi-branch

- skeleton branch
  - input: `skeleton_stream`
  - 舊版：frame MLP + BiLSTM
- location branch
  - input: `location_stream`
  - MLP + BiLSTM
- motion branch
  - input: `motion_stream`
  - MLP + BiLSTM
- fusion
  - concat 三路 embedding
  - fusion MLP
  - classifier

### 第四階段 GCN skeleton branch

- 只替換 skeleton branch
- `skeleton_stream (204)` 先重組成 `(B, T, V, C)`
- `V = 51`
- `C = 4`, 對應 `(x, y, z, mask)`
- 經過 2 層 GraphConv 做空間編碼
- 對 node 做 pooling，得到 `(B, T, hidden)`
- 再交給 BiLSTM 做 temporal encoding
- location / motion branch 與 fusion 保持不變

### 第五階段 ST-GCN skeleton branch

- 只升級 skeleton branch
- 輸入仍來自 `skeleton_stream`
- `204 -> (B, T, 51, 4) -> (B, 4, T, 51)`
- 經過 `STGCNBlock x N`
- 再做 global average pooling + projection head
- 輸出固定為 `(B, 256)`，與 fusion 保持相容

## 各 branch 任務

- skeleton branch：學手型、關節幾何與手部拓樸
- location branch：學手相對於臉、胸、軀幹的位置語意
- motion branch：學速度、方向、加速度、雙手同步性

## 推論控制層

模型輸出之後，不會直接顯示分類結果，而是再經過：

1. valid-hand gate
2. sequence-valid gate
3. confidence threshold
4. temporal smoothing / recent voting
5. final display label

這一層的目的，是讓系統在沒有有效手勢時盡量顯示 `no_sign` 或 `unknown`，而不是硬猜既有類別。

## 與未來升級的關係

- 目前同時支援：
  - `gcn_skeleton_branch.py`
  - `stgcn_skeleton_branch.py`
- 下一步可以把 skeleton branch 升級成：
  - multi-stream GCN
  - graph transformer
- location / motion branch 未來可替換成 attention encoder 或 transformer side stream

## 參考來源

### Realtime-Sign-Language-Detection-Using-LSTM-Model

- `v2/main.py`
  - webcam + MediaPipe Holistic
  - sequence 收集概念
  - 即時推論骨架

### Indian-Sign-Language-Detection

- `dataset_keypoint_generation.py`
  - landmark normalization 拆模組思路

### HA-SLR-GCN

- `Code/Network/SL_GCN/data_gen/sign_gendata.py`
- `Code/Network/SL_GCN/feeders/feeder_27.py`
  - `N,C,T,V,M` 目標格式參考
  - graph-based skeleton encoder 的後續升級方向
