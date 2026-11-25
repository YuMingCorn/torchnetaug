# 專案總結：torchnetaug

**繁體中文** | [English](summary.en.md)

狀態：已結束（2026-11-24）。

## 重點

1. **實作符合論文。** 471 個測試證明套件算出的值等於論文的定義（[design.md 第 9 節](docs/design.md#9-正確性驗證)）。
2. **NetAug 沒有提高 base 模型的測試準確率。** CIFAR-10（3 個 seed）與 CIFAR-100（8 個 seed）的所有設定都在雜訊內、沒有實質增益或無法判定。
3. **baseline 多訓練 40 epoch 更好**：+1.54 個百分點，而且顯著。它的計算量仍少於 NetAug（本環境中每步約 2.6 倍）。
4. **額外的 loss 訊號沒有轉成泛化能力。** 訓練準確率上升（CIFAR-100 ×1 平均 +1.16 個百分點），測試準確率沒有。增強梯度和 base 梯度大致同方向（cosine 0.82 到 0.93），所以 NetAug 在這裡接近放大步長；對照結果是「無法判定」，這只是推測。
5. **結論只適用於本專案的範圍**：CIFAR、TinyResNet、40 epoch。論文的增益來自 ImageNet、MobileNet 類模型、150 epoch，本專案沒有重現那個設定。

## 主要結果

base 模型的測試準確率與同 seed baseline 的配對差異（百分點）。完整結果見 [experiments.md](docs/experiments.md)。

| 設定 | 差異 | 判讀 |
|---|---|---|
| CIFAR-10，寬度增強（3 seed） | +0.15 ± 0.74 | 雜訊內 |
| CIFAR-10，深度增強（3 seed） | −0.05 ± 0.29 | 雜訊內 |
| CIFAR-100 ×0.25，NetAug（8 seed） | −0.14 [−0.76, +0.49] | 沒有實質增益 |
| CIFAR-100 ×0.5，NetAug（8 seed） | +0.46 [−0.61, +1.52] | 無法判定 |
| CIFAR-100 ×1，NetAug（8 seed） | −0.10 [−0.90, +0.69] | 無法判定 |
| CIFAR-100 ×0.25，baseline 多訓練 40 epoch（8 seed） | **+1.54 [+0.85, +2.24]** | 有增益 |

## 未來工作

- **在論文的設定重現**：ImageNet、MobileNetV2-Tiny、150 epoch 加 5 epoch warmup。若只用 32×32 資料，先調整 MobileNet 的步幅。
- **找出 MobileNetV2 變差的原因**：它的增強梯度和 base 梯度的 cosine 只有 0.12 到 0.64。
- **縮小仍為「無法判定」設定的信賴區間**：8 個 seed 已將後半停止增強、每 epoch 排序與全域 ×1/×2 收斂為「沒有實質增益」，但 `--init max`、×0.5、×1 與論文的 weight decay 仍橫跨 0 與 +0.5 門檻，並需事先設計寬路徑檢查的對照組。
- **降低每步成本**：本環境中 NetAug 每步約是 baseline 的 2.6 倍，論文為 +16.7%。
- **補齊套件限制**：DDP 實測、被加寬層上的自訂 buffer、`Conv1d`/`Conv3d`/轉置卷積、無法追蹤的模型（可評估組合模組的方案，見 [design.md 第 11 節](docs/design.md#11-設計決定與限制)）。

## 文件

| 文件 | 內容 |
|---|---|
| [README.md](README.md) | 安裝與 API 用法 |
| [docs/design.md](docs/design.md) | 程式碼設計與正確性驗證 |
| [docs/experiments.md](docs/experiments.md) | 實驗設定、結果與重現指令 |
| [docs/glossary.md](docs/glossary.md) | 中英術語對照 |
| [docs/reference/](docs/reference/) | 論文與官方實作的整理 |
