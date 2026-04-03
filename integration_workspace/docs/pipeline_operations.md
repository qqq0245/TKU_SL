# Pipeline Operations

## 1. 如何跑大規模 landmarks extraction

```bash
python scripts/build_label_map.py
python scripts/scan_video_dataset.py
python scripts/extract_landmarks_batch.py --skip-existing
```

如果你想分批跑：

```bash
python scripts/extract_landmarks_batch.py --offset 0 --limit 500 --skip-existing
python scripts/extract_landmarks_batch.py --offset 500 --limit 500 --skip-existing
```

## 2. 如何續跑

landmarks 與 sequence export 都支援：

- `--skip-existing`
- `--offset`
- `--limit`
- `--class-filter`

也就是你可以把長任務拆成多次執行，不需要重頭開始。

## 3. 如何重試失敗影片

landmarks：

```bash
python scripts/extract_landmarks_batch.py --retry-failed
```

sequence export：

```bash
python scripts/export_sequences_batch.py --retry-failed
```

## 4. 如何查看 pipeline status

```bash
python scripts/pipeline_status.py
```

也可輸出文字或 json：

```bash
python scripts/pipeline_status.py --txt-out dataset_pipeline/logs/status.txt --json-out dataset_pipeline/logs/status.json
```

## 5. 如何看 class coverage report

```bash
python scripts/build_class_coverage_report.py
```

主要看：

- `coverage_status`
- `usable_sequence_count`
- `train_count`
- `val_count`
- `test_count`

若一個類別是 `split_ready`，表示已能進入訓練。

## 6. 如何看 failure report

```bash
python scripts/build_failure_report.py
```

這會整合：

- `missing_or_invalid_videos.csv`
- `landmarks_failed.csv`
- `sequences_failed.csv`

## 7. 如何建立 20 / 50 / 100 類 subset

20 類：

```bash
python scripts/build_subset_experiment.py --min-usable-sequences 10 --max-classes 20 --sort-by usable_sequence_count --output-name top20_ready
```

50 類：

```bash
python scripts/build_subset_experiment.py --min-usable-sequences 10 --max-classes 50 --sort-by usable_sequence_count --output-name top50_ready
```

100 類：

```bash
python scripts/build_subset_experiment.py --min-usable-sequences 10 --max-classes 100 --sort-by usable_sequence_count --output-name top100_ready
```

如果想要每類更平均：

```bash
python scripts/build_subset_experiment.py --min-usable-sequences 10 --max-classes 20 --balanced-per-class --output-name top20_balanced
```

## 8. 如何判斷現在能不能進入全量訓練

至少先檢查：

1. `pipeline_status.py` 的 `split ready classes`
2. `class_coverage_report.csv` 裡有多少類是 `split_ready`
3. `failure_summary.json` 是否仍有大量錯誤
4. `low_sample_classes.csv` 是否太多

建議節奏：

1. 先 `20` 類
2. 再 `50` 類
3. 再 `100` 類
4. 最後才考慮完整 `1999` 類
