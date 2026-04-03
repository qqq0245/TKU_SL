# 手語辨識 GitHub 專案分析報告

更新日期：2026-03-18

## 1. 專案總表

| 專案 | 技術類型 | 主要任務 | 使用技術 | 可重用性評分 | 與你的專題關聯性 | 建議用途 |
|---|---|---|---|---:|---|---|
| Realtime-Sign-Language-Detection-Using-LSTM-Model | LSTM + MediaPipe Holistic baseline | 單詞級時序辨識、即時 webcam 推論 | MediaPipe Holistic, OpenCV, TensorFlow/Keras LSTM, NumPy | 5 | 非常高，最貼近「手型+動作+時序」的 baseline 起點 | 當最小可行系統 baseline，直接拆 webcam、landmark、sequence、LSTM 流程 |
| Real-Time-Sign-Language-Recognition | 靜態 hand landmark + CNN | 字母/數字即時辨識 | MediaPipe Hands, PyTorch, Pandas, Excel feature table | 3 | 中，高在手型辨識層，低在時序與詞彙層 | 拿手部 landmark 正規化與簡單 realtime UI；不建議做主幹 |
| Indian-Sign-Language-Detection | 靜態 hand landmark + FNN | ISL 靜態字母/手勢分類 | MediaPipe Hands, TensorFlow, OpenCV, CSV keypoints | 3 | 中，適合手型特徵抽取與標準化參考 | 拿單手 landmark 前處理與 CSV 建置；不建議直接沿用模型 |
| Real-Time-Sign-Language | YOLO 物件偵測 | ASL 字母/hello 偵測與 webcam demo | YOLOv5, PyTorch, OpenCV, Roboflow, LabelImg | 2 | 低到中，較偏手勢偵測而非手語時序理解 | 只拿資料標註/偵測概念，不建議當主系統 |
| HA-SLR-GCN | Skeleton GCN | isolated sign recognition | PyTorch, GCN, skeleton graph, feeder/config pipeline | 5 | 非常高，適合你未來的「動作+位置+時序」研究主幹 | 當進階研究架構，尤其是骨架序列建模與資料格式標準化 |
| SLGTformer | Skeleton Transformer | isolated sign recognition | PyTorch, Transformer attention, graph positional encoding, wandb | 4 | 高，適合第三/四版升級研究 | 當 transformer 升級路線，取 temporal/spatial attention 設計 |
| SignSense | Full pipeline + LLM 概念 | isolated sign recognition + sentence generation demo | TensorFlow, MediaPipe Holistic, parquet, TFLite, Gemini API | 4 | 高在 full pipeline 與句子層概念，但 repo 不完整 | 拿 app pipeline、landmark parquet 流程、句子後處理構想 |

### 整體判斷

- 最值得保留做主幹：`Realtime-Sign-Language-Detection-Using-LSTM-Model`, `HA-SLR-GCN`, `SLGTformer`
- 最值得拆模組：`Realtime-Sign-Language-Detection-Using-LSTM-Model`, `Indian-Sign-Language-Detection`, `SignSense`
- 最適合只拿概念：`Real-Time-Sign-Language`

## 2. 各專案詳細分析

---

## 2.1 Realtime-Sign-Language-Detection-Using-LSTM-Model

GitHub：https://github.com/AvishakeAdhikary/Realtime-Sign-Language-Detection-Using-LSTM-Model

### 專案目標

- 用 webcam 即時擷取手、姿態、臉部 landmarks。
- 把每個手語詞彙拆成固定長度 frame sequence。
- 以 LSTM 進行 sequence classification。

### 使用模型

- 根目錄：舊版 notebook 流程。
- `v2/main.py`：三層 LSTM + Dense classifier。
  - `LSTM(64, return_sequences=True)`
  - `LSTM(128, return_sequences=True)`
  - `LSTM(64, return_sequences=False)`
  - `Dense(64) -> Dense(32) -> Dense(num_classes)`

### 輸入資料形式

- MediaPipe Holistic feature vector，總長 1662 維。
- 組成：
  - pose 33 x 4 = 132
  - face 468 x 3 = 1404
  - left hand 21 x 3 = 63
  - right hand 21 x 3 = 63
- 資料夾格式：
  - `MP_Data/<SIGN>/<sequence_id>/<frame_id>.npy`

### 輸出結果

- `model.h5`
- `model_weights.h5`
- 即時 webcam 推論的類別機率條。

### 主要檔案

- `v2/main.py`
- `v2/README.md`
- `v2/requirements.txt`
- `RealTimeSignLanguageDetection.ipynb`
- `Train.ipynb`

### 前處理 / 訓練 / 推論流程

- 前處理：`MediapipeHandler.extract_keypoints`
- 資料蒐集：`DatasetManager.collect`
- dataset 載入：`DatasetManager.load_dataset`
- 訓練：`ModelHandler.train`
- webcam 推論：`InferenceEngine.run`

### 優點

- 非常貼近你的第一版需求。
- 已經把 webcam、landmarks、sequence、LSTM、live inference 串起來。
- `v2/` 版結構明顯比 notebook 版更可重用。
- 能同時保留 hand/pose/face，多模態擴充空間大。

### 限制

- landmarks 直接串接成 1662 維，未明確拆成 handshape branch / location branch / motion branch。
- 類別是使用者手動輸入 signs，不含固定 label ontology。
- 無 gloss、sentence、language model。
- face 特徵占比過高，可能讓模型偏向背景/人臉姿態而非核心手語差異。

### 可直接重用的部分

- webcam 擷取與 OpenCV 顯示流程
- MediaPipe Holistic 偵測
- landmark flatten 與缺失補 0
- sequence 資料夾組裝
- LSTM baseline
- 即時機率可視化 UI

### 可參考但不建議直接使用的部分

- 直接把 face 全量 468 點塞入 baseline
- 單一 softmax 做所有語義決策
- 互動式 CLI 流程當正式系統主介面

### 必須重構後才適合用的部分

- 將 1662 維拆成你要的多分支特徵：
  - handshape
  - location
  - motion
  - temporal order
- label schema 要改成你的詞彙 / 動作單元定義
- dataset manager 要改成可讀你自己的註記格式

### 與你的專題整合建議

- 直接拿來做 V1 baseline。
- 先保留 hand + pose，face 先降維或移除。
- 新增相對位置特徵：
  - 手腕相對鼻尖、下巴、肩膀、胸口中心
- 新增 motion 特徵：
  - frame difference
  - velocity / displacement
- 後續再把 LSTM 替換成 GCN 或 Transformer。

### 可重用性評分

- 5/5

---

## 2.2 Real-Time-Sign-Language-Recognition

GitHub：https://github.com/MonzerDev/Real-Time-Sign-Language-Recognition

### 專案目標

- 用 MediaPipe Hands 做手部 landmark 偵測。
- 對字母 A-Z 或數字手勢做即時分類。

### 使用模型

- `Sign Language Recognition/CNNModel.py`
- 1D Conv CNN，輸入 shape 約為 `(batch, 63, 1)`。
- 最後輸出 26 類。

### 輸入資料形式

- 21 個手部 landmarks，各自 x/y/z，共 63 維。
- 經 Min normalization：
  - `lm.x - min(x_coords)`
  - `lm.y - min(y_coords)`
  - `lm.z - min(z_coords)`
- 中介資料存在 Excel：
  - `alphabet_data.xlsx`
  - `numbers_data.xlsx`

### 輸出結果

- `CNN_model_alphabet_SIBI.pth`
- `CNN_model_number_SIBI.pth`
- 即時畫面上的字母/數字預測

### 主要檔案

- `Sign Language Recognition/realTime.py`
- `Sign Language Recognition/training.py`
- `Sign Language Recognition/CNNModel.py`
- `Sign Language Recognition/handLandMarks.py`
- `Sign Language Recognition/mediapipeHandDetection.py`

### 優點

- 手型辨識層很直觀。
- landmark normalization 流程簡單，容易抽成模組。
- 附現成權重。
- 即時推論檔短小，容易理解。

### 限制

- 僅單手靜態分類，沒有 body location、motion、temporal modeling。
- 以 Excel 當訓練資料格式，不利大規模專案維護。
- `static_image_mode=True` 用在 webcam，不是最佳即時設定。
- 只適合字母/數字級任務，不是手語詞彙級 sequence 任務。

### 可直接重用的部分

- MediaPipe Hands webcam 擷取
- hand landmark normalization
- bounding box 顯示與簡單即時推論 UI

### 可參考但不建議直接使用的部分

- Excel 當主要 dataset format
- 1D CNN 當你專題的主模型
- 直接沿用 A-Z 類別定義

### 必須重構後才適合用的部分

- 前處理輸出改成 CSV / parquet / npy
- 改成雙手支援與 body anchor features
- 把靜態分類改成 sequence-compatible encoder

### 與你的專題整合建議

- 把它當手型辨識子模組參考。
- 可以先獨立做一個 `handshape_classifier`，再與時序模型融合。

### 可重用性評分

- 3/5

---

## 2.3 Indian-Sign-Language-Detection

GitHub：https://github.com/MaitreeVaria/Indian-Sign-Language-Detection

### 專案目標

- 將 ISL image dataset 轉成 landmark CSV。
- 使用 FNN/簡單 classifier 做即時辨識。

### 使用模型

- README 說明為 feedforward neural network。
- `ISL_classifier.ipynb` 為訓練 notebook。
- `isl_detection.py` 載入 `model.h5` 即時推論。

### 輸入資料形式

- MediaPipe Hands 21 點。
- 只取 2D x/y，相對於 wrist 基準點。
- flatten 為 42 維。
- 存入 `keypoint.csv`。

### 輸出結果

- `model.h5`
- webcam 畫面中的預測 label

### 主要檔案

- `dataset_keypoint_generation.py`
- `isl_detection.py`
- `ISL_classifier.ipynb`
- `keypoint.csv`

### 優點

- 相對座標正規化做得清楚。
- `calc_landmark_list` / `pre_process_landmark` 很適合抽模組。
- CSV 型資料管線比 Excel 好維護。

### 限制

- 仍然是靜態單幀分類。
- 沒有 z-depth、雙手互動、body location、時序。
- 使用固定資料夾結構與檔名假設，泛用性不足。

### 可直接重用的部分

- landmark 轉相對座標
- flatten + normalization
- CSV logging pipeline
- webcam + MediaPipe Hands 推論流程

### 可參考但不建議直接使用的部分

- 單一 FNN classifier 當主架構
- 將字母/手勢視為單幀決策

### 必須重構後才適合用的部分

- 加入 pose/body anchor
- 擴充成雙手 + 時序
- CSV schema 要改成你的 sign unit 與 metadata

### 與你的專題整合建議

- 很適合拿來實作 `hand_landmark_extraction` 的輕量版。
- 若你想先建立「手型辨識層」，此 repo 的 normalization 邏輯可直接借用。

### 可重用性評分

- 3/5

---

## 2.4 Real-Time-Sign-Language

GitHub：https://github.com/paulinamoskwa/Real-Time-Sign-Language

### 專案目標

- 用 YOLOv5 做 ASL 字母即時偵測。
- 也示範自製 `HELLO` 小資料集。

### 使用模型

- YOLOv5 custom detector。
- 訓練 notebook 內直接呼叫：
  - `python train.py --img 448 --batch 64 --epochs 500 ...`

### 輸入資料形式

- 影像 + YOLO bounding box labels
- `American Sign Language (ASL) dataset/data.yaml`
  - `train: ../train/images`
  - `val: ../valid/images`
  - `nc: 26`
  - `names: ['A' ... 'Z']`

### 輸出結果

- `best.pt` 類型的 YOLO 權重
- webcam 即時偵測畫面

### 主要檔案

- `README.md`
- `Yolov5 - ASL dataset/part 1 - Real-Time ASL Detection - Training.ipynb`
- `Yolov5 - ASL dataset/part 2 - Real-Time ASL Detection - Testing.ipynb`
- `American Sign Language (ASL) dataset/data.yaml`

### 優點

- 資料標註與 detection pipeline 示範完整。
- README 很誠實指出 dataset domain mismatch 問題。
- 適合做手部區域粗定位或 ROI 擷取。

### 限制

- 物件偵測不等於手語辨識。
- 缺少 sequence modeling。
- 無 body position / temporal order / language layer。
- 對字母 bbox 偵測效果高度依賴資料來源與拍攝條件。

### 可直接重用的部分

- LabelImg 標註工作流
- YOLO 資料格式
- webcam 偵測展示概念

### 可參考但不建議直接使用的部分

- 直接以 YOLO 做手語主分類
- 把字母 detection 當詞彙辨識

### 必須重構後才適合用的部分

- 若要用，只建議當手部 ROI detector 前級。
- 後面仍要接 landmark/skeleton/temporal model。

### 與你的專題整合建議

- 可用於資料蒐集前的 hand crop pipeline。
- 不建議投入主幹整合優先序。

### 可重用性評分

- 2/5

---

## 2.5 HA-SLR-GCN

GitHub：https://github.com/snorlaxse/HA-SLR-GCN

### 專案目標

- 以 hand-aware skeleton GCN 做 isolated sign recognition。
- 支援 AUTSL / INCLUDE 等資料集。

### 使用模型

- `Code/Network/SL_GCN/model/hand_aware_sl_lgcn.py`
- TCN + GCN block 疊代
- 額外引入：
  - hand-aware adjacency
  - spatial / temporal / channel attention
  - adaptive drop graph

### 輸入資料形式

- shape：`N, C, T, V, M`
- 例：`(28142, 3, 150, 27, 1)`
- 27 點骨架版本，來自 133 點 skeleton 中挑選：
  - body anchor
  - left hand subset
  - right hand subset

### 輸出結果

- train/val/test score
- checkpoint
- work_dir logs

### 主要檔案

- `Code/Network/SL_GCN/main_base.py`
- `Code/Network/SL_GCN/model/hand_aware_sl_lgcn.py`
- `Code/Network/SL_GCN/data_gen/sign_gendata.py`
- `Code/Network/SL_GCN/data_gen/gen_bone_data.py`
- `Code/Network/SL_GCN/data_gen/gen_motion_data.py`
- `Code/Network/SL_GCN/feeders/feeder_27.py`
- `Code/Network/SL_GCN/config/sign_cvpr_A_hands/AUTSL/train_joint_autsl.yaml`

### 優點

- 非常適合你的「手部動作 + 身體位置 + 時序順序」需求。
- 有清楚的 skeleton dataset schema。
- 支援 joint / bone / motion 多種輸入。
- `sign_gendata.py` 的 27 點選點策略很值得參考。

### 限制

- 工程較研究型，整合門檻高。
- 依賴外部 preprocessing data。
- 即時 webcam 不在 repo 主流程中。
- 仍是 isolated recognition，不含句子生成與 LLM。

### 可直接重用的部分

- skeleton sequence 資料格式 `N,C,T,V,M`
- joint / bone / motion 三種特徵生成
- feeder normalization / random mirror
- graph-based temporal modeling 架構

### 可參考但不建議直接使用的部分

- 直接照搬其資料路徑與 AUTSL/INCLUDE label 設計
- 直接沿用 work_dir/config 命名，不利你自己的研究管理

### 必須重構後才適合用的部分

- 把 `sign_gendata.py` 改成讀你自己的 landmarks / pose json / npy
- 自定義 27 點或更多節點配置
- 加入 location-specific branch 或相對位置 channel

### 與你的專題整合建議

- 這是你第二階段最值得接上的研究骨幹。
- 你可以先用 MediaPipe 產生 skeleton，再轉成它的 `N,C,T,V,M` 格式。
- 若你的手語詞彙以「位置+運動+順序」為核心，這個 repo 非常適合當 V3/V4 主架構。

### 可重用性評分

- 5/5

---

## 2.6 SLGTformer

GitHub：https://github.com/neilsong/SLGTformer

### 專案目標

- 以 transformer/attention 架構做 skeleton-based sign recognition。
- 使用 WLASL 預處理骨架資料。

### 使用模型

- `model/twin_attention.py`
- `model/attention.py`
- spatial attention + temporal attention
- graph relative positional encoding
- TwinSVT 式 temporal attention block

### 輸入資料形式

- 同樣是 skeleton sequence，基本 shape 為 `N,C,T,V,M`
- config 預設：
  - `num_point: 27`
  - `window_size: 120`
  - `num_class: 2000`

### 輸出結果

- checkpoints
- test scores
- wandb logs

### 主要檔案

- `main.py`
- `model/twin_attention.py`
- `model/attention.py`
- `feeders/feeder.py`
- `config/WLASL/train/train_joint.yaml`
- `environment.yml`

### 優點

- attention 與 transformer 升級路線清楚。
- feeder 與骨架格式延續 GCN 系統，方便互換。
- 有 Graph Relative Positional Encoding 概念，可延伸到手部位置建模。

### 限制

- 即時 demo 不在 repo 重點。
- 研究導向，較不適合作為第一版系統。
- 依賴外部資料與 pretrained model 下載。

### 可直接重用的部分

- skeleton feeder
- temporal attention block
- graph-relative positional encoding 概念
- transformer classifier 主幹

### 可參考但不建議直接使用的部分

- 直接使用 WLASL 2000 類設定
- 直接照搬全部訓練超參數

### 必須重構後才適合用的部分

- dataset class / label space
- node set 與位置特徵定義
- 訓練成本與 batch size 要依你設備重估

### 與你的專題整合建議

- 在你完成 LSTM baseline 與基本 location features 後，再升級到此架構最合理。
- 可與 HA-SLR-GCN 共用同一份 skeleton sequence 格式。

### 可重用性評分

- 4/5

---

## 2.7 SignSense

GitHub：https://github.com/DEV-D-GR8/SignSense

### 專案目標

- 用 transformer-based ASL model 做 isolated word recognition。
- 透過 Gemini API 將識別出的詞組裝成句子。

### 使用模型

- notebook 中有自定義 Transformer：
  - `MultiHeadAttention`
  - `Transformer`
- `app.py` 使用 `model.tflite` 做推論。

### 輸入資料形式

- `app.py` 使用 MediaPipe Holistic 擷取每 frame 543 landmarks。
- 寫入 parquet，再載入為 `n_frames x 543 x 3`。
- notebook 顯示主要關注：
  - lips
  - left hand
  - right hand
  - pose subset

### 輸出結果

- 即時顯示 sign name
- 累積 `unique_signs`
- 經 Gemini API 生成句子

### 主要檔案

- `app.py`
- `ASL_Model_Training.ipynb`
- `requirements.txt`
- `10042041.parquet`

### 優點

- 很符合你未來的 full pipeline 規劃。
- 已把 landmark sequence、TFLite inference、詞彙累積、LLM 後處理串在一起。
- parquet 管線比 Excel/散落 npy 更工程化。

### 限制

- repo 不完整：`model.tflite`、`train.csv` 不在 repo 內。
- `unique_signs` + LLM 排句仍偏 demo heuristic，不是嚴格的 gloss decoding。
- app 每 3 秒做一次整段預測，對連續手語切分仍不足。

### 可直接重用的部分

- MediaPipe Holistic -> parquet sequence pipeline
- landmark dataframe 組裝方式
- TFLite 推論介面設計
- LLM 句子後處理概念

### 可參考但不建議直接使用的部分

- 直接讓 LLM 自由重排詞序，缺少 deterministic decoding
- `unique_signs` 去重後再排句，會丟失時序與重複詞資訊

### 必須重構後才適合用的部分

- 補齊模型檔與 label mapping
- 把 `unique_signs` 改成 time-stamped token stream
- 增加 sign segmentation / confidence filtering
- LLM 介面改成可插拔後端

### 與你的專題整合建議

- 很適合作為 V4 句子生成層概念來源。
- 不適合直接當你當前主 baseline，因為 repo 缺關鍵檔且前段模型不可直接驗證。

### 可重用性評分

- 4/5

## 3. 技術分層建議

| 技術層 | 目標 | 優先參考 repo | 理由 |
|---|---|---|---|
| 手型辨識層 | 辨識單手/雙手手型、指型、相對手指姿態 | `Real-Time-Sign-Language-Recognition`, `Indian-Sign-Language-Detection` | 兩者都提供輕量 hand landmark normalization 與靜態分類思路 |
| 身體位置辨識層 | 手相對頭、下巴、肩膀、胸口的位置建模 | `Realtime-Sign-Language-Detection-Using-LSTM-Model`, `SignSense`, `HA-SLR-GCN` | 前兩者有 holistic，後者有 body+hand skeleton 選點邏輯 |
| 動作時序辨識層 | frame sequence、motion、順序 | `Realtime-Sign-Language-Detection-Using-LSTM-Model`, `HA-SLR-GCN`, `SLGTformer` | 從 LSTM baseline 到 GCN/Transformer 升級路線完整 |
| 多模態融合層 | handshape + location + motion branch | `Realtime-Sign-Language-Detection-Using-LSTM-Model`, `HA-SLR-GCN`, `SLGTformer` | 前者便於先拆 branch，後兩者適合研究級融合 |
| 詞彙辨識層 | isolated sign classification | `Realtime-Sign-Language-Detection-Using-LSTM-Model`, `HA-SLR-GCN`, `SLGTformer` | 各自代表 baseline / GCN / Transformer |
| 句子生成層 | gloss / sentence / LLM postprocess | `SignSense` | 目前這批 repo 中唯一明確有 LLM 串接概念 |

## 4. 最終整合建議

### 最適合做 baseline 的專案組合

- `Realtime-Sign-Language-Detection-Using-LSTM-Model`
- `Indian-Sign-Language-Detection`

理由：

- 前者提供完整 sequence + realtime baseline。
- 後者可補強 hand landmark normalization 與簡單標準化流程。

### 最適合做進階研究版本的專案組合

- `HA-SLR-GCN`
- `SLGTformer`
- `SignSense`

理由：

- GCN 與 Transformer 可共用 skeleton sequence 資料格式。
- SignSense 提供句子層與部署層概念。

### 最適合做 demo 的專案組合

- `Realtime-Sign-Language-Detection-Using-LSTM-Model`
- `SignSense`

理由：

- 第一個解決 realtime isolated sign demo。
- 第二個提供 sentence/LLM 層展示方向。

### 哪些專案適合只拿概念不拿程式

- `Real-Time-Sign-Language`
  - 拿 detection 與 dataset mismatch 的教訓。
- `Real-Time-Sign-Language-Recognition`
  - 拿 hand-only baseline 概念，程式耦合與資料格式不適合直接主用。

## 5. 你的專題對應結論

### 最接近你目前需求的 repo

1. `Realtime-Sign-Language-Detection-Using-LSTM-Model`
2. `HA-SLR-GCN`
3. `SLGTformer`

### 最值得你直接拿來改的程式段

1. `v2/main.py` 中的 holistic + sequence + realtime inference
2. `dataset_keypoint_generation.py` / `pre_process_landmark` 的 landmark 正規化
3. `sign_gendata.py` + `gen_motion_data.py` 的 skeleton sequence 標準化
4. `app.py` 的 sequence-to-LLM app pipeline

### 下一步最應該先整合哪兩個 repo

- 第一優先：`Realtime-Sign-Language-Detection-Using-LSTM-Model` + `HA-SLR-GCN`

原因：

- 先用前者快速做出你的 baseline 系統。
- 再把同一套 MediaPipe 輸出轉成後者的 `N,C,T,V,M` 骨架格式，直接接研究升級版。
