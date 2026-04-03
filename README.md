# TKU_SL

淡江大學 AI 系畢業專題，AI 手語輔助辨識。

目前這個 repository 為 `99_docs_analysis` 的 code-only GitHub snapshot，保留：
- 程式原始碼
- 設定檔
- metadata / manifest
- 輕量報告與修補紀錄

刻意排除：
- 本地錄製影片與資料集原始影片
- 模型權重與 checkpoint
- `npz` processed sequence artifacts
- cache 與大型 smoke / runtime 產物

若要從完整本地工作區重新匯出乾淨版，可執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\export_code_only_snapshot.ps1 -Clean
```
