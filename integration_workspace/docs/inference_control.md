# Inference Control

## 1. 為什麼需要 no_sign 類別

若訓練資料只有手勢類別，模型在沒有有效手勢時仍會被迫從既有類別中選一個。

因此 `no_sign` / `idle` 應被視為正式類別。

## 2. 建議收集哪些 no_sign 資料

- 人坐在鏡頭前但不比手語
- 手放下
- 自然姿態
- 只動頭或肩膀
- 手在畫面內自然移動，但不是已知手勢
- 不完整手勢或過渡動作

建議樣本數：

- 至少接近主要類別數量
- 若系統主要使用於即時待機狀態，`no_sign` 可以更多

## 3. 為什麼需要 confidence threshold

softmax 最大值並不代表模型真的有把握。

若最大信心低於門檻，系統應顯示：

- `unknown`
- 或 `low_confidence`

而不是強制輸出已知手勢。

## 4. 為什麼需要 temporal smoothing

逐幀或逐窗推論常會跳動。

使用 recent voting 可以讓結果更穩定：

- 收最近 N 次預測
- 做 majority vote
- 計算該類別平均 confidence

## 5. 為什麼需要 valid-hand / sequence gate

若手根本沒被穩定抓到，模型輸出通常沒有意義。

因此目前加入兩層 gate：

### frame-level gate

- 目前 frame 至少要有：
  - 左手或右手其中之一有效
  - pose 有效

### sequence-level gate

- 一段 sequence 中，有效 frame 比例必須大於門檻
- 否則直接顯示：
  - `no_sign`
  - 或 `insufficient_signal`

## 6. 推論最終決策流程

1. webcam / landmarks
2. feature_builder
3. sequence window
4. model raw prediction
5. confidence threshold
6. valid-hand / sequence gate
7. temporal smoothing
8. final display label

## 7. 目前顯示狀態

目前畫面至少會顯示：

- `Prediction: ...`
- `Raw: ...`
- `Status: ...`

常見狀態：

- `collecting`
- `low_confidence`
- `invalid_frame`
- `insufficient_signal`
- `smoothed`
- `warming_up`

## 8. 目前限制

- `no_sign` 若尚未加入訓練資料，系統主要仍依靠 gate 與 threshold 拒答
- smoothing 目前使用簡單 majority vote
- 沒有做 segment-level sign spotting

## 9. 未來可改善方向

- 真的加入大量 `no_sign` 訓練資料
- 使用 HMM / CTC / Viterbi 做更穩定的時序決策
- 做 sign onset / offset segmentation
- 將 smoothing 改成 confidence-aware voting 或 state machine
