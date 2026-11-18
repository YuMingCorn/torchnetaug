# 論文筆記：Network Augmentation for Tiny Deep Learning（NetAug）

- 作者：Han Cai, Chuang Gan, Ji Lin, Song Han（MIT HAN Lab）
- 發表：ICLR 2022
- 論文：<https://arxiv.org/abs/2110.08890>（HTML 版：<https://ar5iv.labs.arxiv.org/html/2110.08890>）
- 官方程式：<https://github.com/mit-han-lab/tinyml/tree/master/netaug>（MIT 授權，Copyright (c) 2021 Han Cai）
- 論文 PDF 本地副本：[papers/cai2022netaug_arxiv2110.08890v2.pdf](papers/cai2022netaug_arxiv2110.08890v2.pdf)（arXiv v2，CC BY-NC-SA 4.0，見 [papers/README.md](papers/README.md)）

引用格式（取自官方 repo README）：

```bibtex
@inproceedings{cai2022network,
    title={Network Augmentation for Tiny Deep Learning},
    author={Han Cai and Chuang Gan and Ji Lin and Song Han},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=TYw3-OlrRm-}
}
```

> **查證狀態**（2026-11-18）：
> - 第 2、3 節的公式、超參數，以及第 6 節的實驗數字，已對照 arXiv v2 PDF 的原文核對，並註明章節或表號。
> - 第 4、5、7 節的官方程式碼內容，讀自 GitHub 原始檔，但經過摘要工具轉述，**沒有逐行核對**。
> - 標記 ⚠️ 的項目仍未查證。

## 1. 核心想法

- 小模型（tiny model）的問題是**欠擬合（underfitting）**，不是過擬合。
- 所以一般的正則化手段（資料增強、dropout）對小模型幫助有限，甚至有害。
- NetAug 反過來做：**增強模型，而不是增強資料**。把小模型嵌進更大的模型裡，讓小模型在訓練時得到額外的監督訊號。
- 推論時只用小模型，**推論成本不變**。

## 2. 訓練目標

```text
L_aug = L(W_t) + α_1 · L([W_t, W_1]) + ··· + α_i · L([W_t, W_i]) + ···      （論文 Eq. 1）
W_t^{n+1} = W_t^n − η ( ∂L(W_t^n)/∂W_t^n + α · ∂L([W_t^n, W_i^n])/∂W_t^n )   （論文 Eq. 2）
```

- `W_t`：小模型的權重。
- `[W_t, W_i]`：增強模型。它包含小模型的權重，再加上額外的權重 `W_i`。
- `α_i`：增強 loss 的權重。論文所有增強模型用同一個 `α`，實驗中 `α = 1.0`（第 3 節，Eq. 2 下方）。
- **每一步只取樣一個增強模型**。論文說取樣更多個不但增加成本，也會讓準確率變差（第 3 節）。增強部分的權重 `W_i` 也同時被更新。
- **權重共用**：只存最大的增強模型。較小的增強模型與小模型都是它的「前段通道切片」。小模型是前 `w` 個通道。

官方 `train.py` 的實作方式：

1. `set_active(mode="min")`：小模型 forward、算 loss、backward。
2. `set_active(mode="random")`：隨機增強模型 forward、算 loss、backward。
3. 兩次的梯度累加後，做一次 optimizer step。

## 3. 增強哪些維度

- **只增強寬度**，不增強深度。論文的理由是：在 GPU 上，加寬比加深便宜。
- 兩個超參數：
  - 增強倍率 `r`：最大寬度是 `r × w`。
  - 多樣性 `s`（diversity factor）：增強寬度在 `w` 到 `r × w` 之間等距分布。
  - 論文範例：`r=3, s=2` → `[w, 2w, 3w]`。從範例看，`s` 是 `w` 以外的增強寬度個數（第 3 節）。
  - 原文：「Different layers can use different augmentation ratios.」也就是不同層可以用不同倍數。
  - 整個網路的所有運算共用同一個 `r`。
- 官方程式有兩組清單（MobileNet 類模型）：
  - `aug_width_mult_list`：各層輸出通道的寬度倍數。
  - `aug_expand_list`：inverted block 的 expand ratio。
  - 不增強：kernel size、stride、分類器輸出。
- 官方 `tinymbv2.sh`：expand `[1.0, 1.6, 2.2, 2.8]`，width `[1.0, 1.8, 2.6]`，影像大小 144，16 張 GPU。
- **寬度取樣的粒度**：官方只在每個 stage 的第一個 block 取樣寬度。殘差 block 沿用輸入寬度，避免殘差相加時通道數不一致。

## 4. 訓練排程（官方程式）

設 `remaining = n_epochs + warmup - epoch`：

| 條件 | 模式 | 意義 |
|---|---|---|
| 預設 | `random` | 寬度與 expand 都隨機取樣 |
| `remaining <= stop_aug_w_epoch` | `min_w` | 寬度固定為最小，expand 仍取樣 |
| `remaining <= stop_aug_e_epoch` | `min_e` | expand 固定為最小，寬度仍取樣 |
| `remaining <= stop_netaug_epoch` | 關閉 | 只訓練小模型 |

- `configs/netaug.yaml` 預設：`stop_aug_w_epoch: 75`、`stop_aug_e_epoch: 0`、`stop_netaug_epoch: 0`、150 epochs、5 warmup epochs。
- README 說寬度增強在訓練後半段停止。
- ⚠️ `tinymbv2.sh` 實際傳入的 `stop_aug_*` 值沒有查到。

其他訓練設定：

- 論文：batch 2048（16 GPU）、Nesterov SGD momentum 0.9、lr 0.4 cosine、weight decay 4e-5。
- `configs/netaug.yaml`：base lr 0.025（依 GPU 數放大）、label smoothing 0.1。
- 名稱含 `bias` 或 `norm` 的參數不做 weight decay。

## 5. 官方程式的技巧

### 5.1 通道排序

- 每個 epoch 開始時執行（`sort_channel` 開啟時）。
- 重要度 = pointwise conv 權重絕對值的總和，由大到小排序。
- 同一個排列要套用到所有相連的層：pointwise conv 的輸入、depthwise conv 與其 BN、inverted conv 與其 BN、SE 層。
- 目的：讓小模型取到的前段通道是最重要的通道。

### 5.2 BN 處理

- `DynamicBatchNorm2d` 依 active 寬度切片 running 統計值與 affine 參數。
- 評估前，先 `set_active("min")`，再用訓練集的子集合重新估計 BN 平均值與變異數（`reset_bn`），多 GPU 時取平均。

### 5.3 初始化

- 用 `kaiming_uniform`，不用 He fan-out。README 說 fan-out 在大 expand ratio 時效果差。
- Linear 權重用 truncated normal（std 0.02）。BN gamma 初始化為 1。
- ⚠️ 是否把殘差分支最後一個 BN 的 gamma 設為 0，沒有查證。

### 5.4 匯出

- 載入 `best.pt` → `set_active("min")` → 跑一次假輸入 → `model.export()` → 存成 `target.pt`。
- `export()` 建出一般的 `MobileNetV2`，每個動態層用 `active_state_dict()` 載入切片後的權重。

### 5.5 動態層的 active 寬度

- 官方的動態層從輸入張量推得輸入通道數（`x.shape[1]`），只需要設定輸出寬度。
- 權重切片方式為 `weight[:out, :in]`。

## 6. 實驗結果（已對照 PDF）

- ImageNet，MobileNetV2-Tiny（r144）：150 epochs 時 51.7% → 53.3%（+1.6%，Table 1）。
- 300 epochs：52.4% → 53.7%（+1.3%）。訓練成本 420 → 490 GPU 小時，多 16.7%（Table 3）。
- 同一張 Table 3：Dropout、Mixup、AutoAugment、RandAugment、DropBlock 都讓 MobileNetV2-Tiny 變差（−0.7% 到 −3.7%）。
- Table 1 最大增益：MobileNetV3 w0.35 r160，58.1% → 60.3%（+2.2%）。
- 達到相同準確率，ImageNet 少用 75% 的 epoch，ImageNet-21K-P 少用 83%（Figure 3）。
- 偵測（YoloV3 + 小 backbone）：
  - 在相近 AP50 下，Pascal VOC 少 41% MACs，COCO 少 38% MACs（摘要與 Figure 5）。
  - Table 4（遷移學習，輸入 416）：MobileNetV3 w0.35 的 AP50 在 VOC 為 63.6 → 65.3，在 COCO 為 29.2 → 30.7。
  - 引言寫的是 VOC +3.36%、COCO +1.8%，和 Table 4 的差值（+1.7、+1.5）不同。抽出的文字中找不到引言數字對應的表格，可能來自 Figure 5 的某個解析度。⚠️ 引用時請註明採用哪一處。
- **大模型會變差**：ResNet50 從 76.8% 降到 76.5%（Table 1）。
  - Figure 4：ResNet50 的訓練準確率上升、驗證準確率下降，作者解讀為過擬合會更嚴重。
  - NetAug 針對的是欠擬合的小模型。

## 7. 官方 repo 結構

```text
netaug/
├── train.py, eval.py, setup.py
├── configs/default.yaml, configs/netaug.yaml
├── bash/imagenet/*.sh, bash/transfer/*.sh, bash/eval.sh
├── models/base/        一般模型（mbv2, mbv3, mcunet, proxylessnas, tiny_mbv2）
├── models/netaug/
│   ├── layers.py       DynamicModule, DynamicConv2d, DynamicDepthwiseConv2d, DynamicLinear,
│   │                   DynamicBatchNorm2d, DynamicConvLayer, DynamicDepthwiseConvLayer,
│   │                   DynamicLinearLayer, DynamicDsConvLayer, DynamicSE,
│   │                   DynamicInvertedBlock, DynamicSeInvertedBlock
│   ├── mbv2.py         NetAugMobileNetV2（set_active, sort_channels, export）
│   ├── tiny_mbv2.py    NetAugTinyMobileNetV2
│   └── utils.py        reset_bn, aug_width, sync_width, sort_param, sort_norm, sort_se, sort_channels_inner
└── utils/              criterion, distributed, init, lr_scheduler, metric, misc, profile
```

## 8. 論文、官方程式與本專案參考實作的差異

本專案參考實作指根目錄的 `netaug/`，詳見 [netaug_reference_impl.md](netaug_reference_impl.md)。

| 項目 | 論文 / 官方程式 | 參考實作 `netaug/` |
|---|---|---|
| 任務與模型 | ImageNet 分類（MobileNet 類）；偵測用 YOLOv3 風格頭 | YOLOv8 偵測（ultralytics） |
| 增強維度 | 寬度 + expand ratio | 只有寬度（`max_depth` 未使用） |
| 寬度清單 | 絕對倍數清單，如 `[1.0, 1.8, 2.6]` | `linspace(1, max_width, num_points) / max_width` |
| 取樣粒度 | 每個 stage 第一個 block 取樣，殘差 block 沿用 | **每一層各自**隨機取樣；`exclude` 類別固定 base 寬度 |
| 每步增強模型數 | 1 個 | 1 個 |
| Loss | `L(base) + 1.0 × L(aug)`，兩次 backward 累加梯度 | `netaug_1.py`：同論文，但一次 backward；`netaug.py`：base 對 aug 做 KD + `L(aug)` |
| KD | 無 | 有（`netaug.py`） |
| 預訓練最大模型 | 無 | 有（`netaug_1.py` 的 `aug_pretrain`） |
| 停止增強 | 依剩餘 epoch，分寬度 / expand / 全部 | 依絕對 epoch `stop_width_aug` |
| 通道排序 | 每個 epoch，依 pointwise 權重；同一排列套到所有相連層 | 每 50 epoch，依 L1 範數（選用）；**沒有重排下游層輸入維度，排序後網路的運算結果會改變** |
| BN 重設 | 評估前，用訓練子集 | 驗證前，用整個訓練集（選用） |
| active 輸入通道 | 從 `x.shape[1]` 推得 | 明確設定 `active_in_channels`，不一致就丟錯 |
| 分組卷積 | 有 `DynamicDepthwiseConv2d` | 不支援（TODO） |
| 匯出 | `set_active("min")` → `export()` | `set_base()` → `export_module()`，每次存檔都匯出 |
