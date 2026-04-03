$ErrorActionPreference = "Stop"

$projectRoot = "c:\Users\qqq02\Desktop\99_docs_analysis"
$workspaceRoot = Join-Path $projectRoot "integration_workspace"
$artifactRoot = Join-Path $workspaceRoot "artifacts_webcam9_relative_coord_v1"
$sourcePipelineRoot = Join-Path $workspaceRoot "dataset_pipeline_webcam9_nosign_seq30s5_iso"
$targetPipelineRoot = Join-Path $workspaceRoot "dataset_pipeline_webcam9_relative_coord_v1"
$recordedRoot = Join-Path $projectRoot "datasets\recorded\webcam_30_words"
$trainingVariantRoot = Join-Path $projectRoot "datasets\training_variants\webcam_9_with_nosign_raw"
$dataDir = Join-Path $targetPipelineRoot "processed_sequences"
$trainSplit = Join-Path $targetPipelineRoot "splits\train.csv"
$valSplit = Join-Path $targetPipelineRoot "splits\val.csv"
$manifestCsv = Join-Path $targetPipelineRoot "manifests\video_manifest.csv"
$labelMapJson = Join-Path $targetPipelineRoot "manifests\label_map.json"
$indexJsonl = Join-Path $dataDir "index.jsonl"
$hardNegativeDir = Join-Path $recordedRoot "no_sign"

$env:SIGN_ARTIFACTS_DIR = $artifactRoot
$env:SIGN_DATASET_PIPELINE_ROOT = $targetPipelineRoot

Write-Host "[INFO] SIGN_ARTIFACTS_DIR=$env:SIGN_ARTIFACTS_DIR"
Write-Host "[INFO] SIGN_DATASET_PIPELINE_ROOT=$env:SIGN_DATASET_PIPELINE_ROOT"
Write-Host "[INFO] Recorded webcam root: $recordedRoot"
Write-Host "[INFO] Training variant root: $trainingVariantRoot"
Write-Host "[INFO] Training data: $dataDir"
Write-Host "[INFO] Train split: $trainSplit"
Write-Host "[INFO] Val split: $valSplit"

if (-not (Test-Path $targetPipelineRoot)) {
  New-Item -ItemType Directory -Path $targetPipelineRoot | Out-Null
}

foreach ($subdir in @("manifests", "landmarks_cache", "logs")) {
  $src = Join-Path $sourcePipelineRoot $subdir
  $dst = Join-Path $targetPipelineRoot $subdir
  if (-not (Test-Path $dst)) {
    Write-Host "[INFO] Copying $subdir from $src"
    Copy-Item -LiteralPath $src -Destination $dst -Recurse
  }
}

if (-not (Test-Path $labelMapJson)) {
  $sourceLabelMap = Join-Path $sourcePipelineRoot "manifests\label_map.json"
  if (-not (Test-Path $sourceLabelMap)) {
    throw "Missing label map: $sourceLabelMap"
  }
  Write-Host "[INFO] Copying label map from $sourceLabelMap"
  Copy-Item -LiteralPath $sourceLabelMap -Destination $labelMapJson -Force
}

if (-not (Test-Path $hardNegativeDir)) {
  Write-Host "[WARNING] Hard-negative directory not found: $hardNegativeDir"
  Write-Host "[WARNING] Training will proceed without newly recorded no_sign clips."
} else {
  $hardNegativeCount = (
    Get-ChildItem -LiteralPath $hardNegativeDir -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension.ToLower() -in @(".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v") }
  ).Count
  Write-Host "[INFO] Recorded hard-negative clips found: $hardNegativeCount"
}

if ((Test-Path $trainingVariantRoot) -and ($trainingVariantRoot -like "*webcam_9_with_nosign_raw")) {
  Write-Host "[INFO] Rebuilding training variant root: $trainingVariantRoot"
  Remove-Item -LiteralPath $trainingVariantRoot -Recurse -Force
}

python "$workspaceRoot\scripts\build_webcam_nosign_dataset.py" `
  --source-root "$recordedRoot" `
  --output-root "$trainingVariantRoot" `
  --explicit-nosign-dir "$hardNegativeDir"

Write-Host "[INFO] Rebuilding video manifest from training variant root"
python "$workspaceRoot\scripts\scan_video_dataset.py" `
  --source-root "$trainingVariantRoot" `
  --label-map-json "$labelMapJson"

Write-Host "[INFO] Extracting landmarks for manifest videos"
python "$workspaceRoot\scripts\extract_landmarks_batch.py" `
  --manifest-csv "$manifestCsv" `
  --skip-existing

if (-not (Test-Path $dataDir)) {
  New-Item -ItemType Directory -Path $dataDir | Out-Null
}
if (-not (Test-Path (Split-Path $trainSplit))) {
  New-Item -ItemType Directory -Path (Split-Path $trainSplit) | Out-Null
}

if (Test-Path $indexJsonl) {
  Write-Host "[INFO] Removing old exported sequences index: $indexJsonl"
  Remove-Item -LiteralPath $indexJsonl -Force
}
Get-ChildItem -LiteralPath $dataDir -Filter *.npz -File -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host "[INFO] Re-exporting 301-dim sequences from cached landmarks"
python "$workspaceRoot\scripts\export_sequences_batch.py" `
  --manifest-csv "$manifestCsv" `
  --sequence-length 30 `
  --stride 5

Write-Host "[INFO] Rebuilding splits against re-exported sequences"
python "$workspaceRoot\scripts\build_splits.py" `
  --index-jsonl "$indexJsonl" `
  --output-dir (Split-Path $trainSplit)

python "$workspaceRoot\train_multibranch.py" `
  --data-dir "$dataDir" `
  --train-split-csv "$trainSplit" `
  --val-split-csv "$valSplit"
