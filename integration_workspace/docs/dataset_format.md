# Dataset Format

## 目前格式

```text
data/processed/
├─ index.jsonl
├─ labels.json
├─ <sample_id>.npz
└─ ...
```

## `.npz` 內容

- `sequence`
  - shape: `(T, F)`
- `class_label`
- `sample_id`
- `metadata_json`

## 單幀 feature vector

組成順序：

1. landmarks features
2. location features
3. motion features

實際是否包含第 2、3 段，由 `feature_mode` 控制。

### 節點數

- left hand: 21
- right hand: 21
- pose upper-body: 9

### landmarks 維度

- landmarks: `(21 + 21 + 9) * 3 = 153`
- masks: `21 + 21 + 9 = 51`
- landmarks stream 總計：`204`

### location 維度

- 13 組相對向量：`13 * 3 = 39`
- 左右手 zone encoding：`7 * 2 = 14`
- validity flags：`4`
- location stream 總計：`57`

### motion 維度

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
- motion stream 總計：`30`

### 總維度

- `landmarks_only`: `204`
- `landmarks_plus_location`: `261`
- `landmarks_plus_location_motion`: `291`

## metadata 最少欄位

- `label`
- `sample_id`
- `sequence_length`
- `feature_dim`
- `created_at`
- `capture_source`
- `normalization_origin`
- `pose_indices`
- `feature_mode`
- `feature_spec`

## no_sign / idle 類別

- `no_sign` 是正式類別，不是保留字
- 可以直接透過：

```bash
python run_capture.py --label no_sign --num-sequences 20 --sequence-length 30
```

建議 `no_sign` 樣本數至少與主要手勢類別相近。

## 設計理由

- 先用 `.npz + index.jsonl` 打通 baseline
- 後續可平滑升級到 parquet / graph tensor / multi-branch schema
- `feature_spec` 可讓 training / inference / export 共用同一個 index map
