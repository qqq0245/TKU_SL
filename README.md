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
