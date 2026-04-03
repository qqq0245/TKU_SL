param(
    [string]$Destination = "E:\99_docs_analysis_github",
    [switch]$Clean
)

$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)

if (Test-Path $destinationRoot) {
    if (-not $Clean) {
        throw "Destination already exists: $destinationRoot. Use -Clean to replace it."
    }
    Remove-Item -LiteralPath $destinationRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null

$includeExtensions = @(
    ".py", ".ps1", ".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".toml", ".sh"
)

$topLevelFiles = @(
    "AGENTS.md",
    "PROJECT_INDEX.md",
    ".gitignore",
    "realtime_tuning_settings.py",
    "realtime_webcam_sign_lang.py",
    "collect_want_teacher_father.py",
    "analyze_repos.py",
    "clone_all.sh"
)

$reportFiles = @(
    "reports/continuous_inference_repair_progress.md",
    "reports/continuous_inference_decode_consistency.json",
    "reports/continuous_inference_decode_consistency_holistic.json",
    "reports/realtime_webcam_runner_last_summary.json"
)

$copyRoots = @(
    @{
        Path = "integration_workspace"
        ExcludeDirs = @("__pycache__", "data", "artifacts", "artifacts_30_full", "artifacts_30_overlap15", "artifacts_30_overlap5")
        ExcludeDirPrefixes = @("artifacts_", "dataset_pipeline", ".pytest_cache")
    },
    @{
        Path = "metadata"
        ExcludeDirs = @()
        ExcludeDirPrefixes = @()
    },
    @{
        Path = "scripts"
        ExcludeDirs = @("__pycache__")
        ExcludeDirPrefixes = @()
    },
    @{
        Path = "99_docs_analysis"
        ExcludeDirs = @()
        ExcludeDirPrefixes = @()
    }
)

function Ensure-ParentDirectory {
    param([string]$Path)
    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Copy-RelativeFile {
    param([string]$RelativePath)
    $sourcePath = Join-Path $sourceRoot $RelativePath
    if (-not (Test-Path $sourcePath)) {
        return
    }
    $targetPath = Join-Path $destinationRoot $RelativePath
    Ensure-ParentDirectory -Path $targetPath
    Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
}

function Should-ExcludePath {
    param(
        [string]$FullName,
        [string[]]$ExcludeDirs,
        [string[]]$ExcludeDirPrefixes
    )

    $segments = $FullName -split "[\\/]"
    foreach ($segment in $segments) {
        if ($ExcludeDirs -contains $segment) {
            return $true
        }
        foreach ($prefix in $ExcludeDirPrefixes) {
            if ($segment.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        }
    }
    return $false
}

foreach ($relativeFile in $topLevelFiles + $reportFiles) {
    Copy-RelativeFile -RelativePath $relativeFile
}

foreach ($entry in $copyRoots) {
    $rootPath = Join-Path $sourceRoot $entry.Path
    if (-not (Test-Path $rootPath)) {
        continue
    }

    Get-ChildItem -Path $rootPath -Recurse -File | Where-Object {
        $extension = $_.Extension.ToLowerInvariant()
        $includeExtensions -contains $extension
    } | Where-Object {
        -not (Should-ExcludePath -FullName $_.FullName -ExcludeDirs $entry.ExcludeDirs -ExcludeDirPrefixes $entry.ExcludeDirPrefixes)
    } | ForEach-Object {
        $relativePath = $_.FullName.Substring($sourceRoot.Length + 1)
        Copy-RelativeFile -RelativePath $relativePath
    }
}

$readmePath = Join-Path $destinationRoot "README.md"
$readmeContent = @'
# 99_docs_analysis GitHub snapshot

This snapshot keeps the program source, configuration, manifests, and lightweight reports only.

Excluded on purpose:
- local videos and captures
- raw training videos
- model weights and checkpoints
- processed `.npz` sequence artifacts
- cache directories and generated smoke-run folders

To rebuild a fresh snapshot from the full local workspace:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\export_code_only_snapshot.ps1 -Clean
```
'@

Set-Content -LiteralPath $readmePath -Value $readmeContent -Encoding UTF8

Write-Output "Snapshot exported to $destinationRoot"
