# torchnetaug 驗證與實驗報告

**繁體中文** | [English](report.en.md)

日期：2026-11-19。術語對照見 [glossary.md](glossary.md)。

## 摘要

本報告回答四個問題，並記錄驗證方法、實驗結果與找到的問題。

1. **NetAug 有沒有深度增強？** 論文、官方程式（MIT HAN Lab）與參考實作 `netaug/` 都只增強寬度。參考實作有 `max_depth` 參數，但沒有被使用。本套件新增了深度增強（`NetAugModel(..., depth={...})`），做法參考同一研究團隊的 Once-for-All 的 elastic depth。
2. **可以直接用 Dynamic 模組建構網路嗎？** 可以手動設定寬度，輸出與普通網路完全相同。但這樣要自己處理每一層的寬度與通道排序。建議的方式是用一般的 `torch.nn` 層、照 PyTorch 原本的方式自行組合網路，再交給 `NetAugModel`：寬度切換、通道排序與匯出都由通道分析自動處理，與模型種類無關。含有動態層的模型現在會被明確拒絕。
3. **匯出後可以轉成 ONNX 嗎？** 可以。base、×2 與隨機寬度的匯出模型，用 dynamo 與 TorchScript 兩種匯出器都能轉換，onnxruntime 的輸出與 PyTorch 一致。
4. **效果如何？** RESULTS_SUMMARY

## 1. 環境

| 項目 | 版本 |
|---|---|
| 程式碼 | 分支 `feat/torchnetaug`（實驗開始時為 commit `941b37a`；深度實驗在本報告的 commit 上執行） |
| Python | 3.14.3 |
| PyTorch | 2.11.0+cu128（另以 2.15.0 CPU 版跑過全部測試） |
| ONNX | onnx 1.23.2、onnxscript 0.7.2、onnxruntime 1.32.0 |
| 硬體 | Windows 10、RTX 3060 Ti 8 GB（驅動 576.02） |

## 2. 正確性驗證

`uv run pytest` 共 434 個測試，全部通過。測試在 float64 下以很緊的容差比對，而且先把所有參數與 BN 統計值隨機化，避免預設值掩蓋錯誤。

| 測試檔 | 數量 | 驗證的事 |
|---|---|---|
| `test_correctness.py` | 80 | base 模式等於原始模型、匯出、排序保持最大寬度函數、排序後有序且冪等、optimizer 狀態重排、BN 不被增強網路污染、梯度 |
| `test_random_architectures.py` | 61 | 60 個隨機產生的架構（殘差、concat、depthwise、SE 等組合）上的同樣性質，含排序後有序且冪等 |
| `test_whitelist_lemmas.py` | 172 | 排序正確性證明的前提：白名單運算逐通道獨立、動態層的排列引理 |
| `test_widths.py` | 55 | 每個寬度的參數量、權重形狀與輸出，等於直接用該寬度建立的普通模型 |
| `test_depth.py` | 42 | 深度增強（第 5 節） |
| `test_onnx.py` | 12 | ONNX 轉換（第 7 節） |
| `test_dynamic_usage.py` | 8 | 手動使用動態層，以及對含動態層模型的拒絕（第 6 節） |
| `test_docs.py` | 4 | 文件中的 Python 範例都能執行（中文與英文版） |

## 3. CIFAR-10 寬度增強實驗

### 3.1 設定

- 模型：TinyResNet（CIFAR 版 ResNet-18，寬度為 1/8：8、16、32、64 通道，176,402 個參數）。
- 訓練：40 epochs、batch 128、Nesterov SGD（momentum 0.9）、lr 0.1 cosine、weight decay 5e-4（BN 與 bias 不做）、隨機裁切與水平翻轉、bf16 autocast。
- NetAug：`width_mults=(1.0, 2.0, 3.0)`、`alpha=1.0`、每個群組各自抽寬度。`+sort10` 表示每 10 個 epoch 排序一次通道。
- 評估：所有模型都先用 20 個訓練批次（共 10,000 張，無資料增強）重新估計 BN 統計值。測試準確率用 10,000 張測試影像，訓練準確率用前 10,000 張訓練影像。
- 每種設定 3 個 seed（0、1、2）。同一個 seed 的 base 權重初始化與資料順序相同，所以可以逐 seed 配對比較。
- 對照組：同一架構在 ×1.5、×2.0、×2.5、×3.0 寬度下單獨訓練，用來和從 NetAug 模型匯出的同寬度模型比較。

### 3.2 結果

WIDTH_RESULTS

### 3.3 解讀

WIDTH_DISCUSSION

## 4. 通道排序

排序的設計保證：最大寬度的網路函數不變，只改變較小寬度保留哪些通道。

排序後的有序性由 `tests/conftest.py` 的 `assert_sorted` 檢查：每個群組重新計算的重要度必須在捨入誤差內非遞增，而且立刻再排序一次時，只能交換重要度在捨入誤差內相同的通道。重新計算的值不是逐位元相同，因為其他群組的排列會改變讀取者權重的加總順序（float32 實測最大約 1.6e-6 相對誤差）；所以檢查使用相對容差（float64 測試為 1e-9，float32 建議 1e-5）。這個檢查套用於 8 個固定模型、60 個隨機架構與 4 個深度增強模型；float32 的 CPU 與 CUDA 上也另外驗證過。把排序改成遞增或隨機排列時（突變測試），檢查都會失敗。

實驗中每次排序都記錄兩件事。

- **最大寬度輸出的變化**：9 次排序（3 個 seed × 第 10、20、30 epoch）中最大為 8.6e-6，而 logit 的絕對值約 20 到 31。這是 float32 加總順序造成的誤差，代表排序在實際訓練的權重上也保持函數不變。
- **移入 base 的通道數**：9 次都是 0。也就是說，排序從來沒有改變 base 模型使用的通道。

原因是結構性的。加寬時，新通道的權重用預設初始化建立，而 PyTorch 的預設初始化依輸入通道數縮放：最大寬度的層輸入通道是原本的 3 倍，所以新權重的標準差較小（例：`layers.0.conv1` 的 base 區塊約 0.070，新區塊約 0.039）。未訓練時，base 通道的重要度已經是任何新通道的 1.06 到 1.19 倍；訓練時 base 通道每一步都會得到梯度，而新通道只在被抽到時才得到梯度，差距維持住了。

所以在這個設定下，排序只改變中間寬度（例如 ×2）保留哪些通道（未訓練時約 170 個通道改變），對 base 模型沒有作用。`+sort10` 與 `netaug` 的 base 準確率差異應視為 seed 雜訊，不是排序的效果。

## 5. 深度增強

### 5.1 設計

詳細設計見 [architecture.md 第 7 節](architecture.md)。重點：

- `depth={"layers.1": 1, ...}`：在指定的 block 後面加入 `k` 個額外 block。增強網路從 `0..k` 隨機抽要執行幾個，base 網路一個都不執行，所以 base 與 `export()` 仍是原始模型。
- 額外 block 是該 block 的副本，權重重新初始化。殘差 block 的副本最後一個 BN 歸零，一開始是恆等函數。
- 額外 block 的輸出和原 block 的輸出屬於同一個通道群組。這一點對非殘差 block 是必要的：否則排序時，block 之後的層在 base 模式與加深模式下會讀到不同排列的通道，網路函數會在沒有錯誤訊息的情況下改變。測試把這個機制關掉時，非殘差網路的排序測試會失敗。
- 深度增強可以和寬度增強同時使用；每一步同時抽寬度與深度。

### 5.2 驗證

`tests/test_depth.py` 在 TinyResNet、torchvision ResNet-18、MobileNetV2 與一個非殘差卷積堆疊上驗證：

- base 模式（訓練與評估模式）與 `export()` 的輸出等於原始模型，`export()` 的 state_dict 鍵也相同；
- 殘差副本一開始是恆等函數；非殘差副本不會被歸零；
- 排序前後，最大寬度下每一種深度組合（包含全部為 0）的輸出不變；
- `export(sizes, depths)` 是普通 PyTorch 模型（沒有任何本套件的模組），輸出等於動態網路，參數量等於 `num_active_params()`；
- `mode_ctx`、`export`、`recalibrate_bn` 結束後還原寬度與深度；`recalibrate_bn(depths=...)` 會更新額外 block 的 BN；
- `netaug_step` 會訓練額外 block，且增強網路不改 BN 統計值；
- 形狀改變、執行兩次、沒有參數、路徑不存在、巢狀等錯誤都有明確訊息。

### 5.3 CIFAR-10 結果

在 TinyResNet 每個 stage 的最後一個 block 之後加 1 個額外 block（`--depth 1`），訓練設定與第 3 節相同。

DEPTH_RESULTS

## 6. 自行組合網路與 Dynamic 模組

### 6.1 建議的方式：普通層 + 自動分析

使用者用一般的 `torch.nn` 層，照 PyTorch 原本的方式寫 `__init__` 與 `forward`，再交給 `NetAugModel`。`torch.fx` 追蹤模型後，分析器求出通道群組：哪些層讀同一組通道、concat 後每一段來自哪個群組、殘差兩邊必須一致。寬度切換、排序與匯出都依照這個結果進行，所以使用者不需要寫任何「這種 block 要怎麼排序」的程式碼。README 的「自行組合網路」範例（concat、depthwise、殘差）由 `tests/test_docs.py` 執行；60 個隨機架構的測試也驗證了這一點。

### 6.2 直接使用動態層

`DynamicConv2d` 等動態層可以單獨使用：手動設定 `active_out` 與 `set_in_segments()` 後，輸出與載入相同切片權重的普通網路完全相同（`test_hand_built_dynamic_network_equals_plain_network`，差異為 0）。但這樣使用者必須自己維持每一層寬度的一致性，而通道排序需要知道所有相連的張量，手動維護很容易出錯。

把含動態層的模型交給 `NetAugModel` 時，原本的行為有兩個問題：

- 經由追蹤時，`torch.fx` 會失敗或把動態層當成未知運算。
- 經由手動 `graph=` 時，模型被接受，但動態層不會被加寬，而且 `export()` 的結果仍含動態層，部署時必須安裝本套件。這與「匯出模型不依賴本套件」的設計相矛盾。

現在 `analyze()` 與 `NetAugModel` 遇到動態層時，會丟出指出模組路徑與類別的 `TypeError`，並說明應傳入普通層組成的模型。

### 6.3 尚未採用的替代方案

另一種做法是提供套件自己的組合模組（例如 `Sequential`、`Residual`、`Concat`），由組合結構直接宣告通道關係，不需要 `torch.fx`。優點是可以支援無法追蹤的模型；缺點是使用者要改用另一套 API，而且大部分功能和現在的自動分析重複。目前無法追蹤的模型可以用自訂 tracer 或手動 `ChannelGraph` 處理，所以暫不採用。若需要支援大量無法追蹤的模型，可以再評估。

## 7. ONNX

`tests/test_onnx.py` 對 TinyResNet 與 MobileNetV2（×0.35）的匯出模型，在 base、×2 與隨機寬度下，分別用 dynamo 與 TorchScript 匯出器轉成 ONNX，再用 onnxruntime（CPU）執行，與 PyTorch 的輸出比對（容差 rtol 1e-4、atol 1e-5，實際差異小於 2e-8）。12 個組合全部通過。

- 匯出模型不含任何本套件的模組，所以 ONNX 圖中只有標準運算。
- `NetAugModel` 本身在 base 模式也能轉成 ONNX，但圖中帶著最大寬度的權重。部署時應先 `export()`。
- ONNX 相關套件是開發依賴，不是套件依賴；沒有安裝時這些測試會跳過。

## 8. 發現並修正的問題

| 問題 | 影響 | 修正 |
|---|---|---|
| `export(sizes)` 建立新層時沒有沿用模型的 dtype | bfloat16（或與預設 dtype 不同）的模型匯出後無法執行 | 新層轉成原層的 dtype 與裝置；`test_export_sizes_keeps_model_dtype` |
| 含動態層的模型被靜默接受 | 不會被加寬，匯出結果依賴本套件 | 明確的 `TypeError`；`test_dynamic_usage.py` |
| 排序從不把通道移入 base | `+sort` 對 base 模型沒有作用 | 不是程式錯誤，是初始化尺度造成的結構性結果；在本報告與文件中說明（第 4 節） |
| 深度增強點是單一層（不是 block）時，`export(sizes, depths)` 失敗 | 只有 base `export()` 能用 | 名稱對應補上 `P.block`、`P.extra.i` 本身；`test_bare_layer_as_depth_point_exports` |
| `depth=` 與 `recalibrate_bn` 只還原根模組的 train/eval 狀態 | 凍結（eval）的子模組被改成訓練模式 | 逐模組保存並還原；`test_depth_keeps_per_module_train_flags` |
| `recalibrate_bn` 在切換狀態時出錯不會還原；base 校正會清掉未執行的額外 block 的 BN 統計值 | 寬度與深度狀態遺失；`update_bn_in_aug=True` 時額外 block 的統計值被清除 | 切換移入 `try`；未執行的額外 block 不重設；`test_recalibrate_bn_restores_state_and_flags_on_error` |
| `NetAugModel._apply` 覆寫了 `nn.Module._apply` | `.to()`、`.cuda()`、`.float()` 等全部失敗 | 改名為 `_apply_state`；`test_module_conversions_work` |
| 深度增強點 block 內被凍結的 BN 變回訓練模式 | 凍結 BN 的 running 統計值會被更新 | `DepthStack` 只設定自己的旗標；`make_extra` 逐模組還原 |
| 深度增強的原型以手寫開關實作 | 實驗結果無法代表套件行為 | 改為套件 API（`depth=`），實驗腳本改用 `examples/cifar_netaug.py --depth` |

## 9. 限制

- 每種設定只有 3 個 seed、40 個 epoch，只用 CIFAR-10 與一個模型。小於約 0.5 個百分點的差異無法和 seed 雜訊區分。
- 每次執行的分鐘數只能大略參考：部分執行期間，同一張顯示卡同時在跑測試或其他程式。
- 深度增強只能用在保持形狀、每次 forward 只執行一次的 block，且需要 `torch.fx` 追蹤。
- 多 GPU（DDP）尚未實際測試。
