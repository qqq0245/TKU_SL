# 下一步整合藍圖

更新日期：2026-03-18

## 核心策略

你的專題不應直接選一個 repo 照抄，而應建立一條清楚的升級路徑：

1. 先用 `LSTM + MediaPipe` 快速完成可跑的單詞辨識 baseline。
2. 再把特徵拆成 `handshape + location + motion`。
3. 之後把資料格式轉成 skeleton sequence，接上 `GCN / Transformer`。
4. 最後才接 gloss / sentence / LLM。

## 第一版：最小可行系統

### 目標

- webcam
- MediaPipe 擷取 hand 與 pose
- 基本特徵抽取
- 簡單時序模型
- 單詞辨識

### 優先參考 GitHub

- 主參考：`Realtime-Sign-Language-Detection-Using-LSTM-Model`
- 補充參考：`Indian-Sign-Language-Detection`

### 要改哪些程式

- 以 `v2/main.py` 為主體，修改：
  - 刪除或降權 face landmarks
  - labels 固定成你的手語詞彙表
  - 資料蒐集流程改為你的專題資料夾
- 參考 `dataset_keypoint_generation.py`：
  - 加入相對座標 normalization 概念

### 建議實作輸出

- `src/capture/holistic_capture.py`
- `src/features/basic_features.py`
- `src/datasets/sequence_dataset.py`
- `src/models/lstm_baseline.py`
- `src/infer/realtime_infer.py`

### 預期成果

- 可錄製 5 到 20 個手語詞彙
- 每詞可做即時辨識
- 可輸出 baseline accuracy 與 confusion matrix

## 第二版：加入位置資訊

### 目標

- 將手的位置相對於頭 / 下巴 / 胸口建模
- 增加 location features
- 與 handshape 結合

### 優先參考 GitHub

- 主參考：`Realtime-Sign-Language-Detection-Using-LSTM-Model`
- 架構參考：`HA-SLR-GCN`
- app 資料流參考：`SignSense`

### 要改哪些程式

- 在 baseline 特徵抽取中新增：
  - 左右手 wrist 相對 nose / chin / shoulders / chest center
  - 手掌中心相對身體 anchor
  - frame-to-frame displacement
- 新增：
  - `location_feature_builder.py`
  - `motion_feature_builder.py`

### 預期成果

- 同手型但不同位置的詞彙可更容易區分
- 可明確分析「位置資訊是否提升辨識率」

## 第三版：加入多分支模型

### 目標

- hand branch
- location branch
- motion branch
- temporal fusion

### 優先參考 GitHub

- baseline 起點：`Realtime-Sign-Language-Detection-Using-LSTM-Model`
- 研究主幹：`HA-SLR-GCN`
- attention 升級預留：`SLGTformer`

### 要改哪些程式

- 將單一 1662 維向量拆成多分支輸入：
  - `handshape_features`
  - `location_features`
  - `motion_features`
- model 改為：
  - branch encoder
  - concat / fusion layer
  - temporal classifier

### 建議模型方向

- 先用簡單版本：
  - hand MLP / CNN encoder
  - location MLP encoder
  - motion LSTM encoder
  - concat 後 softmax
- 再升級成：
  - branch-specific temporal encoder

### 預期成果

- 特徵可解釋性提升
- 模型更符合你的研究問題，而不是黑箱整包 landmarks

## 第四版：研究升級

### 目標

- skeleton GCN 或 transformer
- attention
- gloss / sentence generation

### 優先參考 GitHub

- skeleton GCN：`HA-SLR-GCN`
- transformer：`SLGTformer`
- sentence / LLM：`SignSense`

### 要改哪些程式

- 建立 MediaPipe 到 skeleton graph 的轉換器：
  - 輸出 `N,C,T,V,M`
- 先接 `HA-SLR-GCN`
- 後續切換或並行比較 `SLGTformer`
- 句子層新增：
  - token stream
  - confidence filtering
  - optional LLM postprocess

### 預期成果

- isolated sign accuracy 顯著提升
- 可研究 hand-aware graph、location-aware graph、attention fusion
- 為未來 gloss / sentence generation 打基礎

## 你現在最應該先整合的兩個 repo

### 第一優先

- `Realtime-Sign-Language-Detection-Using-LSTM-Model`
- `HA-SLR-GCN`

### 原因

- 第一個能最快讓你跑出可用 baseline。
- 第二個提供你未來研究版最重要的 skeleton sequence 格式與 graph 架構。
- 這兩者之間的橋接點非常清楚：
  - MediaPipe landmarks
  - sequence window
  - body/hand node selection

## 哪個 repo 最適合當 baseline

- `Realtime-Sign-Language-Detection-Using-LSTM-Model`

理由：

- 即時
- sequence
- LSTM
- easy-to-modify
- 與你的專題需求最直接

## 哪個 repo 最適合做進階研究架構

- 第一選擇：`HA-SLR-GCN`
- 第二選擇：`SLGTformer`

理由：

- `HA-SLR-GCN` 更適合作為你第二、三版研究主幹。
- `SLGTformer` 更像第四版 attention/transformer 升級方案。

## 建議的實作順序

1. 先完成 `MediaPipe Holistic + hand/pose sequence + LSTM baseline`
2. 加入 `relative location features`
3. 將資料輸出成 `N,C,T,V,M`
4. 接上 `HA-SLR-GCN`
5. 再比較 `SLGTformer`
6. 最後再接 `SignSense` 式句子層

## 建議的研究拆分

### 模型層

- `baseline_lstm`
- `multibranch_lstm`
- `gcn_sign_model`
- `transformer_sign_model`

### 特徵層

- `handshape_features`
- `location_features`
- `motion_features`
- `fusion_features`

### 資料層

- `raw_video`
- `holistic_landmarks`
- `sequence_windows`
- `skeleton_graph_sequences`

### 語言層

- `isolated_sign_output`
- `token_stream`
- `gloss_candidate`
- `sentence_postprocess`

## 短期實作清單

### 這一輪最建議你先做

1. 以 `v2/main.py` 改出你自己的 `baseline_capture_train_infer.py`
2. 建立統一資料格式：
   - `sample_id`
   - `label`
   - `frame_idx`
   - `left_hand`
   - `right_hand`
   - `pose`
3. 新增 location features
4. 預留 skeleton graph 轉換腳本

### 若你接下來要我繼續接管，最合理的下一步

- 我直接幫你在這個工作區再建立一個新的整合專案骨架，例如：
  - `sign_language_research/integration_workspace/`
- 然後把：
  - holistic 擷取
  - sequence dataset
  - baseline LSTM
  - location feature builder
  先實作成你自己的可執行版本
