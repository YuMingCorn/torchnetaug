# 專案總結：torchnetaug

**繁體中文** | [English](summary.en.md)

狀態：已結束（2026-11-23）。

## 結論

本專案用純 PyTorch 實作了 Network Augmentation（NetAug，Cai et al., ICLR 2022），並在 CIFAR-10 與 CIFAR-100 上驗證效果。**測試顯示實作符合論文的定義，也沒有找到實作錯誤；但 NetAug 在測試過的設定中沒有帶來顯著增益。** 增強網路提供的額外 loss 訊號常讓訓練準確率上升，卻沒有提高 base 模型的測試準確率。baseline 多訓練 40 epoch 反而顯著較好；本環境中 NetAug 每步約是 baseline 的 2.6 倍時間，所以這個 baseline 用的計算量仍比 NetAug 少。

## 完成的工作

- **套件 `torchnetaug`**（`src/torchnetaug/`，MIT 授權）：任意可被 `torch.fx` 追蹤、由普通 `torch.nn` 層組成的模型都能包裝。通道結構自動分析；寬度與深度增強；以通道群組為單位的排序，在殘差與 concat 網路上保持最大寬度函數不變；匯出回傳原始模型類別，可轉 ONNX。
- **正確性**：471 個測試，證明每個寬度等於同寬度的普通模型、一步訓練的梯度等於論文 Eq. 2、排序不改變最大寬度函數、匯出模型等於動態網路。詳見 [design.md 第 9 節](docs/design.md#9-正確性驗證)。
- **實驗**：CIFAR-10 寬度、深度與排序；CIFAR-100 欠擬合設定的診斷、對照組與官方配方選項；MobileNetV2 初步測試。詳見 [experiments.md](docs/experiments.md)。

## 主要結果

base 模型的測試準確率與同 seed baseline 的配對差異（百分點）：

| 設定 | 差異 | 判讀 |
|---|---|---|
| CIFAR-10，寬度增強（3 seed） | +0.15 ± 0.74 | 雜訊內 |
| CIFAR-10，寬度增強 + 排序（3 seed） | +0.34 ± 0.44 | 雜訊內 |
| CIFAR-10，深度增強（3 seed） | −0.05 ± 0.29 | 雜訊內 |
| CIFAR-100 ×0.25，NetAug（8 seed） | −0.14 [−0.76, +0.49] | 沒有實質增益 |
| CIFAR-100 ×1，NetAug（5 seed） | −0.46 [−1.05, +0.14] | 沒有實質增益 |
| CIFAR-100 ×0.25，官方配方選項與論文的 weight decay（各 5 seed） | −0.25 到 +0.32，區間都包含 0 | 無法判定 |
| CIFAR-100 ×0.25，baseline 多訓練 40 epoch（8 seed） | **+1.54 [+0.85, +2.24]** | 有增益 |
| CIFAR-100 MobileNetV2 0.35 + warmup（1 seed） | −4.06 | 變差，原因未確定 |

## 觀察

1. **額外的 loss 訊號沒有轉成泛化能力。** 訓練準確率上升（CIFAR-10 每個 seed +0.7 到 +1.3 個百分點），測試準確率沒有。
2. **增強梯度大致和 base 梯度同方向**（TinyResNet 上 cosine 0.82 到 0.93），NetAug 在這裡的作用接近放大步長。但「只是放大步長」的對照結果是「無法判定」，所以這是推測。
3. **排序幾乎不會改變 base。** 新增通道的初始化較小，base 通道又每步都有梯度，所以 base 通道一直是最重要的。
4. **匯出的寬模型不能取代單獨訓練的寬模型**（低 1.3 到 20 個百分點）。
5. **實驗範圍和論文不同。** 論文的增益來自 ImageNet、MobileNet 類模型、150 epoch。官方預設還會在最後 75 個 epoch 停止寬度增強，並每個 epoch 排序。本專案沒有重現那個設定，所以結論只適用於小資料集、短訓練與 TinyResNet。

## 未來工作

- **在論文的設定重現**：ImageNet、MobileNetV2-Tiny、150 epoch 加 5 epoch warmup。若只用 32×32 資料，先調整 MobileNet 的步幅。
- **找出 MobileNetV2 變差的原因**：它的增強梯度和 base 梯度的 cosine 只有 0.12 到 0.64。
- **為「無法判定」的設定增加 seed**：`--init max`、`--stop-aug`、×0.5、論文的 weight decay，並為寬路徑檢查事先設計對照組。
- **降低每步成本**：本環境中 NetAug 每步約是 baseline 的 2.6 倍，論文為 +16.7%。
- **套件限制**：DDP 實測；被加寬層上的自訂 buffer；`Conv1d`、`Conv3d`、轉置卷積；無法追蹤的模型（可評估組合模組的方案，見 [design.md 第 11 節](docs/design.md#11-設計決定與限制)）。

## 文件

| 文件 | 內容 |
|---|---|
| [README.md](README.md) | 安裝與 API 用法 |
| [docs/design.md](docs/design.md) | 程式碼設計與正確性驗證 |
| [docs/experiments.md](docs/experiments.md) | 實驗設定、結果、觀察與重現指令 |
| [docs/glossary.md](docs/glossary.md) | 中英術語對照 |
| [docs/reference/](docs/reference/) | 論文、官方實作與參考實作的整理 |
