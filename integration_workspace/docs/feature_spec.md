# Feature Spec

## Feature mode

- `landmarks_only`
- `landmarks_plus_location`
- `landmarks_plus_location_motion`

## landmarks stream

- 維度：`214`
- 組成：
  - left hand wrist-relative xyz: `63`
  - right hand wrist-relative xyz: `63`
  - pose mid-shoulder-relative xyz: `27`
  - left hand mask: `21`
  - right hand mask: `21`
  - pose mask: `9`
  - explicit finger states: `10`
    - left thumb~pinky: `5`
    - right thumb~pinky: `5`

## location stream

- 維度：`57`

### relative vectors

- left_to_nose: `3`
- left_to_chin: `3`
- left_to_mouth: `3`
- left_to_shoulder_center: `3`
- left_to_chest_center: `3`
- left_to_torso_center: `3`
- right_to_nose: `3`
- right_to_chin: `3`
- right_to_mouth: `3`
- right_to_shoulder_center: `3`
- right_to_chest_center: `3`
- right_to_torso_center: `3`
- left_to_right: `3`

合計：`39`

### zone encoding

- left zones: `7`
- right zones: `7`

合計：`14`

### validity

- left hand valid
- right hand valid
- chin valid
- mouth valid

合計：`4`

## motion stream

- 維度：`30`
- left velocity: `3`
- right velocity: `3`
- left speed: `1`
- right speed: `1`
- hand distance delta: `1`
- left acceleration: `3`
- right acceleration: `3`
- left torso-relative motion: `3`
- right torso-relative motion: `3`
- left direction: `3`
- right direction: `3`
- sync score: `1`
- validity flags: `2`

## 最終維度

- `landmarks_only`: `214`
- `landmarks_plus_location`: `271`
- `landmarks_plus_location_motion`: `301`

## index 區間

### landmarks_only

- landmarks: `0:214`

### landmarks_plus_location

- landmarks: `0:214`
- location: `214:271`

### landmarks_plus_location_motion

- landmarks: `0:214`
- location: `214:271`
- motion: `271:301`

## stream 對應

- skeleton_stream
  - index: `0:214`
  - dim: `214`
- location_stream
  - index: `214:271`
  - dim: `57`
- motion_stream
  - index: `271:301`
  - dim: `30`

## multi-branch 取用方式

- skeleton branch 讀 `skeleton_stream`
- location branch 讀 `location_stream`
  - 訓練時可套用 branch dropout
- motion branch 讀 `motion_stream`
- 最後 concat 三個 branch embedding 做 fusion classifier

## GCN skeleton branch 取用方式

- 輸入來源仍是 `skeleton_stream`
- 其中前 `204` 維 graph 區塊仍依固定順序重排成 graph node：
  - `pose (9)`
  - `left_hand (21)`
  - `right_hand (21)`
- 每個 node 的 feature：
  - `x`
  - `y`
  - `z`
  - `mask`
- 追加的 `10` 維 explicit finger states 會在 branch 內做輔助投影，再與 graph embedding 融合
- reshape 後 tensor:
  - `(B, T, 51, 4)`

## GCN graph 對應

- node count: `51`
- node feature dim: `4`
- adjacency shape: `(51, 51)`
- normalized adjacency shape: `(51, 51)`

## 缺失值處理

- 缺失 landmarks：補 `0`
- 缺失 hand：該手相關 location / motion 向量補 `0`
- 第一幀 motion：補 `0`
- face ref 缺失：chin / mouth 相關向量補 `0`
- 所有缺失會對應 mask / validity flags
