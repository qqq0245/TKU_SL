# 可重用模組整理

更新日期：2026-03-18

## 模組 1：`realtime_webcam_pipeline`

- 來源專案：
  - `Realtime-Sign-Language-Detection-Using-LSTM-Model`
  - `SignSense`
- 對應檔案：
  - `01_baseline_lstm/Realtime-Sign-Language-Detection-Using-LSTM-Model/v2/main.py`
  - `06_full_pipeline_or_llm/SignSense/app.py`
- 依賴套件：
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `tensorflow` 或 `tflite`
- 可否直接搬移：
  - 可直接搬 70%
- 需要修改重點：
  - 統一 camera loop 與 frame buffer
  - 分離 UI 與模型推論
  - 支援你自己的 label map 與 confidence rule
- 優先採用版本：
  - V1 用 `Realtime-Sign-Language-Detection-Using-LSTM-Model/v2/main.py`
  - V4 demo 可加上 `SignSense/app.py` 的文字輸出區

## 模組 2：`hand_landmark_extraction`

- 來源專案：
  - `Indian-Sign-Language-Detection`
  - `Real-Time-Sign-Language-Recognition`
- 對應檔案：
  - `02_hand_landmark_or_static_sign/Indian-Sign-Language-Detection/dataset_keypoint_generation.py`
  - `02_hand_landmark_or_static_sign/Real-Time-Sign-Language-Recognition/Sign Language Recognition/handLandMarks.py`
  - `02_hand_landmark_or_static_sign/Real-Time-Sign-Language-Recognition/Sign Language Recognition/mediapipeHandDetection.py`
- 依賴套件：
  - `mediapipe`
  - `opencv-python`
  - `numpy`
  - `pandas`
- 可否直接搬移：
  - 可直接搬 80%
- 需要修改重點：
  - 改成雙手同時輸出固定 schema
  - 保留 handedness
  - 把輸出從 Excel 改為 CSV / parquet / npy
- 優先採用版本：
  - `Indian-Sign-Language-Detection` 的相對座標 normalization 較乾淨

## 模組 3：`holistic_landmark_extraction`

- 來源專案：
  - `Realtime-Sign-Language-Detection-Using-LSTM-Model`
  - `SignSense`
- 對應檔案：
  - `01_baseline_lstm/Realtime-Sign-Language-Detection-Using-LSTM-Model/v2/main.py`
  - `06_full_pipeline_or_llm/SignSense/app.py`
- 依賴套件：
  - `mediapipe`
  - `opencv-python`
  - `numpy`
  - `pandas`
- 可否直接搬移：
  - 可直接搬 85%
- 需要修改重點：
  - face landmarks 可先裁減
  - 補上 relative position features
  - 統一輸出為 `frame x node x channel`
- 優先採用版本：
  - 若你要簡單 baseline，採 `v2/main.py`
  - 若你要 parquet / dataframe 形式，採 `SignSense/app.py`

## 模組 4：`landmark_normalization`

- 來源專案：
  - `Indian-Sign-Language-Detection`
  - `Real-Time-Sign-Language-Recognition`
- 對應檔案：
  - `dataset_keypoint_generation.py::pre_process_landmark`
  - `handLandMarks.py`
  - `realTime.py`
- 依賴套件：
  - `numpy`
  - `itertools`
  - `copy`
- 可否直接搬移：
  - 可直接搬 90%
- 需要修改重點：
  - 從單手改成雙手 + body anchor normalization
  - 增加 z-depth 與 scale normalization
- 優先採用版本：
  - `Indian-Sign-Language-Detection/pre_process_landmark`

## 模組 5：`sequence_builder`

- 來源專案：
  - `Realtime-Sign-Language-Detection-Using-LSTM-Model`
  - `SignSense`
- 對應檔案：
  - `v2/main.py::DatasetManager.prepare_folders`
  - `v2/main.py::DatasetManager.load_dataset`
  - `app.py::all_landmarks / parquet accumulation`
- 依賴套件：
  - `numpy`
  - `os`
  - `pandas`
  - `pyarrow`
- 可否直接搬移：
  - 可直接搬 75%
- 需要修改重點：
  - 將資料格式改為你自己的 sign sample metadata
  - 補上 sample id、sign id、start/end frame、body location tags
- 優先採用版本：
  - baseline 用 `v2/main.py`
  - 工程化儲存可參考 `SignSense`

## 模組 6：`lstm_classifier`

- 來源專案：
  - `Realtime-Sign-Language-Detection-Using-LSTM-Model`
- 對應檔案：
  - `01_baseline_lstm/Realtime-Sign-Language-Detection-Using-LSTM-Model/v2/main.py`
- 依賴套件：
  - `tensorflow`
  - `keras`
  - `scikit-learn`
- 可否直接搬移：
  - 可直接搬 80%
- 需要修改重點：
  - 輸入改為 hand/location/motion 分支或降維後特徵
  - 改善 train/val split 與 metrics
  - 加上 model checkpoint / early stopping
- 優先採用版本：
  - 直接採此版本做你第一版 baseline

## 模組 7：`static_handshape_classifier`

- 來源專案：
  - `Real-Time-Sign-Language-Recognition`
  - `Indian-Sign-Language-Detection`
- 對應檔案：
  - `CNNModel.py`
  - `training.py`
  - `ISL_classifier.ipynb`
- 依賴套件：
  - `torch` 或 `tensorflow`
  - `pandas`
  - `numpy`
- 可否直接搬移：
  - 可直接搬 50%
- 需要修改重點：
  - 從字母/數字改成你的 handshape ontology
  - 改成可作為主模型的 branch encoder
- 優先採用版本：
  - 只採概念，不建議原樣照搬

## 模組 8：`skeleton_graph_pipeline`

- 來源專案：
  - `HA-SLR-GCN`
- 對應檔案：
  - `Code/Network/SL_GCN/data_gen/sign_gendata.py`
  - `Code/Network/SL_GCN/data_gen/gen_bone_data.py`
  - `Code/Network/SL_GCN/data_gen/gen_motion_data.py`
  - `Code/Network/SL_GCN/feeders/feeder_27.py`
  - `Code/Network/SL_GCN/config/sign_cvpr_A_hands/*`
- 依賴套件：
  - `torch`
  - `numpy`
  - `pyyaml`
  - `pickle`
- 可否直接搬移：
  - 可直接搬 65%
- 需要修改重點：
  - 改資料來源讀取器
  - 自定義節點選取與 graph adjacency
  - 對齊你的 label / split / metadata
- 優先採用版本：
  - 高優先，作為研究版主幹

## 模組 9：`gcn_sign_classifier`

- 來源專案：
  - `HA-SLR-GCN`
- 對應檔案：
  - `Code/Network/SL_GCN/model/hand_aware_sl_lgcn.py`
  - `Code/Network/SL_GCN/main_base.py`
- 依賴套件：
  - `torch`
  - `tensorboard`
- 可否直接搬移：
  - 可直接搬 60%
- 需要修改重點：
  - 調整 `num_class`
  - 加入你想要的 location branch
  - 重新定義 graph nodes 與 body anchors
- 優先採用版本：
  - 第二優先主模型，適合你第二到四版

## 模組 10：`transformer_sequence_encoder`

- 來源專案：
  - `SLGTformer`
  - `SignSense`
- 對應檔案：
  - `05_transformer_sign/SLGTformer/model/twin_attention.py`
  - `05_transformer_sign/SLGTformer/model/attention.py`
  - `06_full_pipeline_or_llm/SignSense/ASL_Model_Training.ipynb`
- 依賴套件：
  - `torch`
  - `timm`
  - `einops`
  - `tensorflow`
- 可否直接搬移：
  - 可直接搬 50%
- 需要修改重點：
  - 選定單一框架，建議先採 PyTorch 路線
  - 對齊 skeleton format 與 class space
  - 降低訓練成本
- 優先採用版本：
  - `SLGTformer` 優先，工程結構比 notebook 更適合研究整合

## 模組 11：`dataset_schema_for_sign_research`

- 來源專案：
  - `Realtime-Sign-Language-Detection-Using-LSTM-Model`
  - `HA-SLR-GCN`
  - `SignSense`
- 對應檔案：
  - `v2/main.py`
  - `sign_gendata.py`
  - `app.py`
- 依賴套件：
  - `numpy`
  - `pickle`
  - `pandas`
  - `parquet`
- 可否直接搬移：
  - 不建議直接搬，建議整合重設
- 需要修改重點：
  - 建立統一 schema：
    - `sample_id`
    - `sign_label`
    - `frame_idx`
    - `left_hand/right_hand/pose/body_anchor`
    - `location_feature`
    - `motion_feature`
    - `split`
- 優先採用版本：
  - 以 `HA-SLR-GCN` 的 `N,C,T,V,M` 為訓練主格式
  - 以 `SignSense` 的 dataframe/parquet 為採集暫存格式

## 模組 12：`sentence_postprocessing`

- 來源專案：
  - `SignSense`
- 對應檔案：
  - `06_full_pipeline_or_llm/SignSense/app.py::get_display_message_from_api`
- 依賴套件：
  - `google-generativeai`
  - `python-dotenv`
- 可否直接搬移：
  - 可直接搬 40%
- 需要修改重點：
  - 不要只傳 `unique_signs`
  - 改成有順序、信心分數、時間戳的 token list
  - 設計 deterministic prompt 與 fallback rule
- 優先採用版本：
  - 只當概念起點

## 模組 13：`llm_text_generation`

- 來源專案：
  - `SignSense`
- 對應檔案：
  - `app.py`
- 依賴套件：
  - `google-generativeai`
- 可否直接搬移：
  - 可直接搬 30%
- 需要修改重點：
  - 抽象成 `llm_backend.py`
  - 支援 Gemini / OpenAI / local model
  - 加入 prompt guardrails 與 error handling
- 優先採用版本：
  - 不建議現在先整合，等 isolated sign pipeline 穩定後再接

## 模組優先級總結

### 第一優先，建議直接採用並改造

1. `holistic_landmark_extraction`
2. `sequence_builder`
3. `lstm_classifier`
4. `skeleton_graph_pipeline`

### 第二優先，建議做成子模組

1. `hand_landmark_extraction`
2. `landmark_normalization`
3. `gcn_sign_classifier`
4. `transformer_sequence_encoder`

### 第三優先，先保留概念

1. `sentence_postprocessing`
2. `llm_text_generation`
3. `static_handshape_classifier`

## 我對你的建議版本

- 第一版應採：
  - `Realtime-Sign-Language-Detection-Using-LSTM-Model/v2/main.py`
  - 搭配 `Indian-Sign-Language-Detection` 的 normalization 想法
- 第二版應採：
  - 自己新增 `location_feature_builder`
  - 輸出對齊 `HA-SLR-GCN` 格式
- 第三版後應採：
  - `HA-SLR-GCN`
  - 視資源再升級 `SLGTformer`
