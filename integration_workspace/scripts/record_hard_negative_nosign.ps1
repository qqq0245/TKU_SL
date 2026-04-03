$ErrorActionPreference = "Stop"

$projectRoot = "c:\Users\qqq02\Desktop\99_docs_analysis"
$workspaceRoot = Join-Path $projectRoot "integration_workspace"
$outputRoot = Join-Path $projectRoot "datasets\recorded\webcam_30_words"
$manifestCsv = Join-Path $projectRoot "metadata\webcam_hard_negative_nosign_manifest.csv"

Write-Host "[INFO] Output root: $outputRoot"
Write-Host "[INFO] Manifest: $manifestCsv"
Write-Host "[INFO] Recording label: no_sign"
Write-Host "[INFO] Suggested target: 5-10 clips, each 15-30 seconds"

python "$workspaceRoot\run_webcam_data_collection.py" `
  --output-root "$outputRoot" `
  --manifest-csv "$manifestCsv" `
  --labels no_sign `
  --clip-seconds 20 `
  --segment-gap-seconds 1.5 `
  --countdown-seconds 3 `
  --fps 20 `
  --width 960 `
  --height 540
