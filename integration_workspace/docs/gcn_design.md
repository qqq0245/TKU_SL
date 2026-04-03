# GCN Design

## 1. 為什麼需要 GCN

原本的 skeleton branch 會把整段骨架向量當成一般 feature sequence 處理，雖然能學到時序，但不會顯式利用手部與上半身關節之間的拓樸關係。

GCN branch 的目的是：

- 先學空間關節關係
- 再做時間建模
- 讓 skeleton branch 更接近未來 ST-GCN / HA-SLR-GCN 的方向

## 2. skeleton graph 定義

目前 graph 使用 `51` 個 node：

- pose: `9`
- left hand: `21`
- right hand: `21`

節點順序固定為：

1. pose
2. left hand
3. right hand

每個 node feature 為：

- `x`
- `y`
- `z`
- `mask`

因此 skeleton graph tensor 形狀為：

- `(B, T, V, C) = (B, T, 51, 4)`

## 3. node / edge 設計

### pose edges

- nose 與左右肩相連
- 左右肩互連
- 肩膀連到手肘
- 手肘連到手腕
- 肩膀連到髖部
- 左右髖互連

### hand edges

每隻手使用 MediaPipe hand 拓樸：

- wrist 到五指根部
- 各手指內部節點串接
- palm 的橫向連接

### pose-hand bridge

- left pose wrist 連到 left hand wrist
- right pose wrist 連到 right hand wrist

## 4. 模型結構

skeleton GCN branch 目前是簡化版：

1. `204` 維 skeleton stream 重組成 `(B, T, 51, 4)`
2. GraphConv x 2
3. 對 node 維度做平均 pooling
4. 得到 `(B, T, hidden_channels)`
5. 交給 BiLSTM 做 temporal encoding
6. 取最後時間步 representation 當作 skeleton embedding

這樣的設計保留了：

- graph-based spatial modeling
- 與現有 multi-branch fusion 的相容性
- 與舊版 LSTM skeleton branch 的可切換性

## 5. 與 HA-SLR-GCN 的關係

目前版本不是完整的 ST-GCN，也不是直接複製 `HA-SLR-GCN`。

關係如下：

- 已經建立 graph node / edge / normalized adjacency
- 已經把 skeleton stream 轉為 graph tensor
- 已經把 skeleton branch 變成 graph-aware spatial encoder

還沒有做的部分：

- temporal graph convolution
- 多種 adjacency partition
- multi-person `M > 1`
- 完整 `N,C,T,V,M` 訓練管線
- 與 location / motion stream 的更深層 cross-stream fusion

## 6. 與目前 ST-GCN branch 的關係

第五階段已經把 skeleton branch 升級成真正的 ST-GCN 版本，因此本文件可視為：

- graph 定義與 adjacency 的底層說明
- 簡化 GCN branch 的設計說明
- ST-GCN 的前一階段

目前差距主要剩：

- 多 adjacency partition
- 更深的 ST-GCN stack
- `N,C,T,V,M` 批次訓練格式
- multi-stream GCN 融合
