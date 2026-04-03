# Migration To GCN

## baseline sequence 格式

- shape: `(T, F)`
- 每幀包含：
  - landmarks stream
  - location stream
  - motion stream

## 未來 GCN 格式

參考 `HA-SLR-GCN`：

- `N, C, T, V, M`

## 本工作區目前轉法

`src/pipeline/gcn_export_adapter.py`

1. 只取 landmarks stream
2. 把 landmarks stream 解回節點
3. location / motion stream 暫不進骨架張量
4. 節點順序固定：
   - `pose`
   - `left_hand`
   - `right_hand`
5. 輸出：
   - `C, T, V, M`

## 目前節點

### pose

- `0`: nose
- `11`: left_shoulder
- `12`: right_shoulder
- `13`: left_elbow
- `14`: right_elbow
- `15`: left_wrist
- `16`: right_wrist
- `23`: left_hip
- `24`: right_hip

### hands

- left hand: 21 點
- right hand: 21 點

### 總節點數

- `V = 9 + 21 + 21 = 51`

## 與 HA-SLR-GCN 差異

- HA-SLR-GCN 常用 27 點骨架
- 你之後可以：
  - 保留 51 點自建 graph
  - 或壓縮成 27 點以便貼近原 repo

## 目前 multi-stream 對應

- skeleton branch
  - 已可切換成簡化 GCN encoder
  - 已可切換成 ST-GCN encoder
  - 是最接近未來 ST-GCN 的主幹
- location branch
  - 保持 side stream
  - 適合未來做 auxiliary branch 或 cross-stream fusion
- motion branch
  - 仍是 temporal side stream
  - 未來可升級成 TCN / transformer motion encoder

## stream 對應

### skeleton stream

- 來自 normalized landmarks section
- 可轉為 `C,T,V,M`
- 是目前 multi-branch 中最接近未來 GCN branch 的部分

### location stream

- 手相對 nose / chin / mouth / shoulder_center / chest_center / torso_center
- zone encoding
- 不屬於 GCN node 本體，較適合：
  - multi-stream fusion
  - side MLP branch
- 屬於 auxiliary stream

### motion stream

- velocity / speed / acceleration / direction / sync
- 不屬於靜態 graph node，本質上更像 temporal side stream
- 在 multi-branch 中是獨立 motion branch

## 未來對接方向

- `HA-SLR-GCN`
  - 主吃 skeleton stream
  - location/motion 可做外掛分支
- transformer
  - 可直接吃三路 token 或多分支 encoder

## 下一步

1. 先累積 baseline dataset
2. 用目前 `51` 點 graph 驗證 GCN / ST-GCN skeleton branch
3. 匯出 `C,T,V,M`
4. 批次堆疊成 `N,C,T,V,M`
5. 再補 graph adjacency variant、multi-stream fusion 與 label split
