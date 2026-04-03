# Multibranch Design

## 1. 設計目的

把目前的 feature-level fusion baseline，升級成 model-level fusion baseline。

也就是：

- 不再讓單一 LSTM 直接吃完整 291 維向量
- 改由三個 branch 分別學不同語意訊號

## 2. 三個 branch 各自負責什麼

### skeleton / handshape branch

- normalized landmarks
- hand shape
- upper-body geometry
- mask 資訊
- 第四階段可切換成 graph-based encoder

### location branch

- 手相對 nose / chin / mouth / shoulder / chest / torso 的位置
- zone encoding
- valid flags

### motion branch

- velocity
- acceleration
- speed
- direction
- hand distance delta
- sync score

## 3. 輸入維度

- skeleton branch: `204`
- location branch: `57`
- motion branch: `30`

## 4. 模型結構

location branch 與 motion branch 目前仍是：

1. per-frame MLP projection
2. BiLSTM temporal encoder
3. 取最後 hidden representation

skeleton branch 現在有兩種模式：

### LSTM skeleton branch

1. per-frame MLP projection
2. BiLSTM temporal encoder
3. 取最後 hidden representation

### GCN skeleton branch

1. 把 `204` 維 skeleton stream 重組成 `(B, T, 51, 4)`
2. GraphConv x 2
3. node pooling
4. BiLSTM temporal encoder
5. 取最後 hidden representation

### ST-GCN skeleton branch

1. 把 `204` 維 skeleton stream 重組成 `(B, T, 51, 4)`
2. 轉成 `(B, 4, T, 51)`
3. `STGCNBlock x N`
4. global average pooling
5. linear projection 成 skeleton embedding

fusion：

1. concat 三個 branch embedding
2. fusion MLP
3. classifier

## 5. fusion 方式

- `concat(skeleton_repr, location_repr, motion_repr)`
- 經過 `fusion_hidden_dim` 的 MLP
- 輸出類別 logits

## 6. 與單一 LSTM 的差異

### 單一 LSTM

- 優點：簡單
- 缺點：不同語意訊號混在一起，較難解釋

### multi-branch

- 優點：
  - 更符合手語結構
  - 更容易升級成 GCN / Transformer / multi-stream
  - 各 branch 可獨立替換
- 缺點：
  - 模型較複雜
  - 需要明確 feature slicing

## 7. 下一步如何升級到 GCN / Transformer

- skeleton branch
  - 已支援簡化 GCN 與 ST-GCN
  - 下一步可升級成 multi-stream ST-GCN 或 graph transformer
- location branch
  - 可保留 MLP/LSTM
  - 或改成 attention encoder
- motion branch
  - 可替換成 TCN / Transformer
- fusion
  - 可升級成 cross-attention 或 gated fusion

## 8. 推論控制層的角色

即使 multi-branch 模型已能更好地分開學習三種訊號，實際即時系統仍需要控制層來避免亂猜。

目前控制層包含：

- `no_sign` 類別
- confidence threshold
- recent voting / temporal smoothing
- valid-hand / sequence-valid gate
