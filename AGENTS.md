# AGENTS.md

## Project root
`E:\99_docs_analysis`

## Mission
持續修復手語 continuous inference serving path，直到所有高優先 residual 都完成處理並通過最終整體驗證為止。

你不是做完一輪就回報，也不是做完一條主線就停止。  
你必須持續執行、持續更新 progress log、持續從最新 artifact 推進到下一條主線。

---

## Always read first
每次開始前，先讀：

- `reports/continuous_inference_repair_progress.md`
- 最新 `reports/realtime_tests_smoke_poseanchor/*/session_summary.json`
- 最新 `reports/realtime_tests_smoke_poseanchor/*/trigger_segment_alignment.csv`
- 最新 `reports/realtime_tests_smoke_poseanchor/*/*_summary.json`
- 最新 `reports/realtime_tests_smoke/*/*.json`

---

## Locked best-known stack
以下已是最佳已知 stack。除非有明確 regression 證據，否則不要修改其核心邏輯：

- `pose_local_anchor=torso_center`
- `mother` fix
- `you/i` fix
- `you-vs-like` fix
- `like-vs-i` fix
- `father rescue`

目前最佳已知結果：
- best-known stack 可重現 WER `0.375`
- tokens 可重現為：`you / mother / father / student / student / like`

---

## Accepted findings
以下結論已成立，直接承接，不要重做已證偽方向：

### Stable fixes
- `mother` serving 修補已落地，且可穿過 downstream emit
- `you/i` 修補已落地
- `you-vs-like` scorer 修補已落地
- `like-vs-i` scorer 修補已落地
- `father rescue` 已落地

### Downgraded / falsified directions
不要再優先投入以下方向，除非 progress log 有新證據：
- threshold tuning
- margin tuning
- broad feature-window mismatch
- broad face-reference remap
- broad active point switching
- hand-mask suppression
- frontloaded resampling 作為主線
- gesture-profile assist 作為主線
- sampling policy 微調
- 單純 handedness inversion
- 已失敗的 direct feature calibration 方案

### want line current status
`want` 主線目前已收斂為：
- exact-span `want -> no_sign` 的最小 culprit 是 `pose_coords_only`
- `mid_shoulder <-> torso_center` 可重現 `student/no_sign` 翻轉
- 但 live `want` reject segment 上，即使使用相同 `pose_coords_only` counterfactual，也只會從 `no_sign` 變成更強的 `student`
- 不會變成 `want`
- prototype / feature-group 統計指向 `pose_context_graph`
- 不是 `location`
- 不是 `motion`

結論：
- `want` 不是窄 inference-side heuristic 可解
- 需要先測試非常窄的 exact-span `want` pose-context reconstruction override
- 若仍無法讓 `want` 贏過 `student`，就要把 inference-side heuristic 修補降級，並明確標記剩餘問題屬於 training/data-domain

---

## Global task queue
你必須依序持續完成下列任務，不能因為只修好其中一項就停止：

1. 保持 `mother` fix 穩定
2. 保持 `you/i` fix 穩定
3. 保持 `you-vs-like` fix 穩定
4. 保持 `like-vs-i` fix 穩定
5. 保持 `father rescue` 穩定
6. 處理 `want` 主線：
   - 先做 exact-span `want` pose-context reconstruction override 的最小高價值實驗
   - 若失敗，將 inference-side heuristic 修補降級，並明確記錄剩餘問題屬於 training/data-domain
7. 接著處理下一個最高價值 residual：
   - 優先看 `teacher/father -> no_sign/student`
   - 或任何最新 artifact 顯示更高影響度的 residual
8. 最後做一次全流程整體驗證，確認所有已修補項目同時成立，沒有互相回歸

---

## Current mainline selection rule
### 當前主線
先從這條開始：

- `want` broader representation / reconstruction branch

### 若 `want` 線失敗
若 exact-span `want` pose-context reconstruction override 仍無法把 `want` 拉過 `student`：
- 不要停止
- 不要交接
- 直接把 inference-side heuristic 修補降級
- 明確標記殘留問題屬於 training/data-domain
- 然後立即切到下一個最高價值 residual

### 下一主線候選優先順序
1. `teacher/father -> no_sign/student`
2. 最新 artifact 顯示影響 WER / coverage 更高的 residual
3. 其他尚未處理的高優先 serving residual

---

## Continuous execution rule
你不得在每輪完成後暫停。

每一輪結束後，你必須立刻：
1. 更新 `reports/continuous_inference_repair_progress.md`
2. 讀取你剛產生的新 artifact
3. 判斷本輪是否改善
4. 若改善，固定這個修補並立即選出下一個最高價值 residual
5. 若未改善，標記 failed hypothesis、回退或保留最佳版本、縮小主線
6. 直接進入下一輪
7. 不要等待使用者，不要要求確認，不要輸出「完成」「先到這裡」「等你再決定」之類語句

---

## Required agent loop
無限重複以下 loop，直到總停止條件全部滿足：

1. Read current state and progress log
2. Write a short plan
3. Make the smallest necessary code/data-path change
4. Run the relevant validation
5. Save artifacts
6. Update progress log
7. If improved, lock the fix and move to the next highest-value residual
8. If failed, mark failed hypothesis, revert or keep best-known version, narrow the mainline
9. Continue immediately into the next round

---

## Required output every round
每輪都必須輸出並記錄：

1. 修改了哪些檔案
2. 修改了哪些函式或邏輯
3. 執行了哪些命令
4. 產生了哪些 artifact
5. 本輪主線是什麼
6. 本輪最重要的結論是什麼
7. 是否改善
8. 哪些假設被排除或降級
9. 下一輪唯一主線是什麼

---

## Progress log requirements
每輪都要更新：

`E:\99_docs_analysis\reports\continuous_inference_repair_progress.md`

每輪至少寫入：
- 時間
- 本輪主線
- 本輪假設
- 修改檔案
- 執行命令
- artifact 路徑
- 結果摘要
- 是否改善
- 是否證偽某假設
- 下一輪唯一主線

---

## Never stop after
即使發生以下任一情況，你也不得停止：

- 完成一輪分析
- 完成一輪修改
- 跑完一次 smoke rerun
- 產出一批 artifact
- 更新 progress log
- 修好一條主線
- 某條路線失敗
- 某輪沒有改善
- 已經連續執行一段時間
- 覺得目前已經足夠交接
- 你認為可以等使用者回來再說
- 只完成一部分總任務清單

---

## If a round fails
若某輪修補失敗、沒有改善、或造成 regression：

- 不得停止
- 不得交接
- 必須標記 failed hypothesis
- 必須保留或回退到最佳版本
- 必須縮小主線
- 必須直接進入下一輪

---

## Hard constraints
- 不要修改 locked best-known stack，除非有明確 regression 證據
- 不要把 threshold / margin tuning 當主要修補
- 不要重新打開已證偽的 broad branches
- 不要同時開多條主線
- 一次只處理一條最高價值主線
- 若某條主線已證明屬 training/data-domain，不要繼續用 inference-side heuristic 硬修

---

## Stop only when ALL are true
只有在以下全部條件都滿足時，才允許停止：

1. 所有高優先 residual 都已被處理
2. 所有已落地修補同時穩定成立
3. 已完成最終整體驗證
4. 沒有更高優先、未處理的 serving residual
5. 所有證據都已寫入 artifact 與 progress log

---

## Final validation requirement
真正允許停止前，必須做一次全流程整體驗證，至少包含：

- 最新 continuous trigger smoke rerun
- 最新 exact-span / serving parity 對照
- 最新 session summary
- 最新 trigger segment alignment
- 最新 emitted tokens 與 WER 對照
- 各修補項目是否同時成立的總結 artifact

---

## Final output rule
在總停止條件全部滿足前，不得使用以下語句：
- finished
- done
- complete
- stopping
- handoff complete
- ready to stop
- 可以停止
- 已完成
- 可交接

只有在全部高優先任務完成後，才允許輸出最終收尾報告。

---

## When in doubt
如果你在任何時刻覺得：
- 這一輪差不多了
- 先停一下
- 等使用者回來再說
- 先做交接

那代表你違反規則。

正確動作永遠是：
1. 更新 progress log
2. 讀取本輪 artifact
3. 選出下一個最高價值 residual
4. 直接進入下一輪