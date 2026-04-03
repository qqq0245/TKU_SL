# ST-GCN Design

## 1. 為什麼從簡化 GCN 升級到 ST-GCN

簡化 GCN branch 的流程是：

- GraphConv
- GraphConv
- node pooling
- BiLSTM

這樣已經能利用 graph 結構，但 spatial 與 temporal 仍是分開學的。

ST-GCN 的目標是把：

- spatial graph convolution
- temporal convolution
- residual connection

放進同一個 block，讓 skeleton branch 本身就具備空間與時間耦合建模能力。

## 2. ST-GCN block 結構

每個 `STGCNBlock` 包含：

1. `SpatialGraphConv`
2. `BatchNorm2d`
3. `TemporalConv`
4. residual connection
5. `ReLU`

目前版本採用穩定簡化版設計：

- adjacency 使用固定 normalized adjacency
- temporal conv 使用 `Conv2d(kernel=(k,1))`
- residual 可由 config 控制開關

## 3. tensor shape 流程

原始 skeleton stream：

- `(B, T, 204)`

先重組成 graph tensor：

- `(B, T, 51, 4)`

再轉成 ST-GCN 內部格式：

- `(B, 4, T, 51)`

經過 ST-GCN block stack 後：

- `(B, C_hidden, T, 51)`

最後做 global average pooling：

- `(B, C_hidden)`

再經過 projection head：

- `(B, 256)`

## 4. skeleton stream 如何 reshape

`204` 維切法如下：

- left hand xyz: `63`
- right hand xyz: `63`
- pose xyz: `27`
- left hand mask: `21`
- right hand mask: `21`
- pose mask: `9`

重組後 node 順序固定為：

1. pose `9`
2. left hand `21`
3. right hand `21`

每個 node feature：

- `x`
- `y`
- `z`
- `mask`

## 5. 與簡化 GCN branch 的差異

### 簡化 GCN

- 先 graph conv
- 再 node pooling
- 再用 BiLSTM 做 temporal encoding

### ST-GCN

- 每個 block 內同時做 spatial + temporal
- temporal modeling 直接在 graph feature map 上進行
- 用 residual 讓 deeper graph stack 更穩

## 6. 為什麼先做 ST-GCN 而不是直接 Transformer

目前 skeleton stream 已經有明確 graph node 結構，先走 ST-GCN 有幾個好處：

- 能最大化利用已建立的 adjacency
- 與 HA-SLR-GCN 路線更接近
- 參數量較可控
- 對資料量較小時通常比直接上 transformer 穩定

## 7. 下一步如何走向後續版本

下一步可以往三條路發展：

### multi-stream ST-GCN

- skeleton 當主幹
- location / motion 變成 side stream
- fusion 升級成更深層 cross-stream 融合

### Transformer temporal encoder

- 在 skeleton branch 後端加 temporal transformer
- 或直接替換 location / motion branch 的 BiLSTM

### gloss / sentence generation

- 先讓 word-level backbone 穩定
- 再把 frame/clip encoder 接到 sequence decoder
- 後續再做 gloss、LM、sentence generation
