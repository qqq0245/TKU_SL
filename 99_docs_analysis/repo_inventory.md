# GitHub Repo Inventory

更新日期：2026-03-18

## 1. 總覽

| Repo | GitHub | 本地路徑 | 主分類 | 次分類 | Clone 狀態 | README 掃描 | 相依檔掃描 | 主要腳本掃描 | 備註 |
|---|---|---|---|---|---|---|---|---|---|
| Realtime-Sign-Language-Detection-Using-LSTM-Model | https://github.com/AvishakeAdhikary/Realtime-Sign-Language-Detection-Using-LSTM-Model | `sign_language_research/01_baseline_lstm/Realtime-Sign-Language-Detection-Using-LSTM-Model` | `01_baseline_lstm` | `02_hand_landmark_or_static_sign`, `06_full_pipeline_or_llm` | 成功 | 成功 | 成功，`v2/requirements.txt` | 成功，`v2/main.py` + notebooks | 根目錄是舊版 notebook，`v2/` 是較可用的模組化版本 |
| Real-Time-Sign-Language-Recognition | https://github.com/MonzerDev/Real-Time-Sign-Language-Recognition | `sign_language_research/02_hand_landmark_or_static_sign/Real-Time-Sign-Language-Recognition` | `02_hand_landmark_or_static_sign` | `06_full_pipeline_or_llm` | 成功 | 成功 | README 內嵌 requirements，無獨立 `requirements.txt` | 成功，`realTime.py`, `training.py`, `CNNModel.py`, `handLandMarks.py` | 偏靜態字母/數字分類，附 `.pth` 權重 |
| Indian-Sign-Language-Detection | https://github.com/MaitreeVaria/Indian-Sign-Language-Detection | `sign_language_research/02_hand_landmark_or_static_sign/Indian-Sign-Language-Detection` | `02_hand_landmark_or_static_sign` | `01_baseline_lstm` | 成功 | 成功 | 成功，`requirements.txt` | 成功，`isl_detection.py`, `dataset_keypoint_generation.py`, `ISL_classifier.ipynb` | 單手靜態 landmark + FNN 流程清楚 |
| Real-Time-Sign-Language | https://github.com/paulinamoskwa/Real-Time-Sign-Language | `sign_language_research/03_detection_yolo/Real-Time-Sign-Language` | `03_detection_yolo` | `02_hand_landmark_or_static_sign` | 成功 | 成功 | 以 notebook 為主，無專案級 requirements | 成功，train/test notebooks, `data.yaml` | 偏物件偵測，不適合直接當手語時序主幹 |
| HA-SLR-GCN | https://github.com/snorlaxse/HA-SLR-GCN | `sign_language_research/04_skeleton_gcn/HA-SLR-GCN` | `04_skeleton_gcn` | `05_transformer_sign` | 成功 | 成功 | 成功，`requirements` | 成功，`main_base.py`, `hand_aware_sl_lgcn.py`, `sign_gendata.py`, feeders/config | 研究型骨架辨識架構，重用價值高 |
| SLGTformer | https://github.com/neilsong/SLGTformer | `sign_language_research/05_transformer_sign/SLGTformer` | `05_transformer_sign` | `04_skeleton_gcn` | 成功 | 成功 | 成功，`environment.yml` | 成功，`main.py`, `twin_attention.py`, `attention.py`, feeder/config | Transformer 型骨架辨識，偏研究升級版 |
| SignSense | https://github.com/DEV-D-GR8/SignSense | `sign_language_research/06_full_pipeline_or_llm/SignSense` | `06_full_pipeline_or_llm` | `02_hand_landmark_or_static_sign`, `05_transformer_sign` | 成功 | 成功 | 成功，`requirements.txt` | 成功，`app.py`, `ASL_Model_Training.ipynb` | `app.py` 依賴 `model.tflite` 與 `train.csv`，repo 內未提供 |

## 2. 目錄分類確認

```text
sign_language_research/
├─ 01_baseline_lstm/
│  └─ Realtime-Sign-Language-Detection-Using-LSTM-Model/
├─ 02_hand_landmark_or_static_sign/
│  ├─ Real-Time-Sign-Language-Recognition/
│  └─ Indian-Sign-Language-Detection/
├─ 03_detection_yolo/
│  └─ Real-Time-Sign-Language/
├─ 04_skeleton_gcn/
│  └─ HA-SLR-GCN/
├─ 05_transformer_sign/
│  └─ SLGTformer/
├─ 06_full_pipeline_or_llm/
│  └─ SignSense/
└─ 99_docs_analysis/
```

## 3. 掃描依據

各 repo 至少檢查以下項目：

- `README.md` 或 `readme.md`
- `requirements.txt` / `environment.yml` / `setup.py` / notebook 安裝步驟
- 主要訓練程式
- 主要推論程式
- 前處理或資料轉換程式
- 模型定義檔
- config / data yaml / dataset 結構說明
- 是否存在預訓練權重或模型檔

## 4. 重要狀態註記

- `Realtime-Sign-Language-Detection-Using-LSTM-Model`：原 repo 根目錄主要是 notebook；`v2/main.py` 是較新且較易拆模組版本。
- `Real-Time-Sign-Language-Recognition`：資料是 Excel 特徵表，不是影像資料夾直接訓練。
- `Real-Time-Sign-Language`：repo 很大，內含實際 Roboflow / YOLO 格式資料與 `labelImg`，已做靜態分析，未執行訓練。
- `HA-SLR-GCN`：README 指向外部 Google Drive 預處理骨架資料，repo 內未含完整訓練資料。
- `SLGTformer`：README 指向外部 skeleton-aware 預處理資料與外部 pretrained models。
- `SignSense`：`app.py` 需要 `model.tflite` 與 `train.csv`，目前 repo 內未找到；可分析流程，但無法直接執行完整 demo。

## 5. 輔助腳本

- `sign_language_research/clone_all.sh`
  - 可重複 clone 全部 repo。
- `sign_language_research/analyze_repos.py`
  - 可列出各 repo 關鍵檔案與腳本候選，用於後續擴充掃描。
