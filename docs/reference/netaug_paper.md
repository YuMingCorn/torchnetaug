# 論文筆記：Network Augmentation for Tiny Deep Learning（NetAug）

- 作者：Han Cai, Chuang Gan, Ji Lin, Song Han（MIT HAN Lab）
- 發表：ICLR 2022
- 論文：<https://arxiv.org/abs/2110.08890>（HTML 版：<https://ar5iv.labs.arxiv.org/html/2110.08890>）
- 官方實作：<https://github.com/mit-han-lab/tinyml/tree/master/netaug>（MIT 授權，Copyright (c) 2021 Han Cai）
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

> **查證狀態**（2026-11-19）：
> - 第 2 節（官方 `train.py` 部分）、第 3 節（官方清單與取樣粒度）、第 4 節、第 5 節、第 7 節（repo 結構），以及第 8 節，已逐行對照官方實作。對照的版本是 commit `0e58f314dfe12967d6165f45238b1ea77b0f7a14`（2022-06-16，master，淺層 clone）。
> - 第 6 節與第 9 節的論文數字，已對照 arXiv v2 PDF 的文字抽取。圖只看到抽取出的文字，沒有看圖片。
> - 上一版標示 ⚠️ 的三項已解決：
>   1. `tinymbv2.sh` 沒有傳入 `stop_aug_*`，實際值是 yaml 預設（75 / 0 / 0）。
>   2. 殘差連接最後一個 BN 的 gamma **沒有**設為 0。
>   3. 論文 intro 的 VOC +3.36%、COCO +1.8%，來自 Figure 5 的圖上數值。原文寫「找不到」是錯的。
> - 未查證：`models/base/` 的 MobileNetV3、MCUNet、ProxylessNAS 基礎模型（只看了 netaug 的包裝層）；`netaug/figures/*.png`；`bash/transfer/` 的 car 以外的四個腳本（只用 grep）。
> - 論文 p.8 的六個「MACs 對準確率」子圖沒有圖說，本文不指定它們的圖號。

**標記**：
- 【論文】：只有論文寫。
- 【官方】：只有官方實作有。論文沒寫。
- 【兩者】：論文與官方實作都寫。

## 1. 核心想法

- 【論文】小模型（tiny model）的問題是**欠擬合（underfitting）**，不是過擬合（論文 p.1）。
- 【論文】所以一般的正則化手段（資料增強、dropout）對小模型幫助有限，甚至有害（論文 p.1–2，Table 3）。
- 【論文】NetAug 反過來做：**增強網路，而不是增強資料**。把小模型嵌進更大的網路裡，讓小模型在訓練時得到額外的監督訊號。作者把它稱為反向 dropout（reverse dropout）（論文 p.1）。
- 【論文】推論時只用小模型，**推論成本不變**（論文 p.1、§3.3）。

## 2. 訓練目標

```text
L_aug = L(W_t) + α_1 · L([W_t, W_1]) + ··· + α_i · L([W_t, W_i]) + ···      （論文 Eq. 1，p.3）
W_t^{n+1} = W_t^n − η ( ∂L(W_t^n)/∂W_t^n + α · ∂L([W_t^n, W_i^n])/∂W_t^n )   （論文 Eq. 2，p.4）
```

- 【兩者】`W_t`：小模型的權重。
- 【兩者】`[W_t, W_i]`：增強網路。它包含小模型的權重，再加上額外的權重 `W_i`。
- 【論文】`α_i`：增強 loss 的權重。論文所有增強網路用同一個 `α`，實驗中 `α = 1.0`（論文 p.5 lines 241–242）。論文沒有 `α` 的消融實驗。
- 【論文】**每一步只取樣一個增強網路**。論文說取樣更多個會增加成本，也會讓準確率變差（論文 p.5 lines 243–246）。這句話沒有附數據。`W_i` 在該步同時被更新。
- 【論文】權重共用：只存最大的增強網路。較小的增強網路都是它的子網路。
- 【官方】小模型取到的是前段通道。切片寫在 `models/netaug/layers.py:89-94`（`weight[:out, :in]`）。小模型的 `set_active("min")` 取 `min(out_channels_list)`（`models/netaug/mbv2.py:181-182`）。

官方 `train.py` 的實作方式【官方】：

1. `set_active(mode="min")`：小模型 forward、算 loss、backward（`train.py:126-133`）。
2. `set_active(mode=...)`：增強網路 forward、算 loss、backward（`train.py:135-142`）。模式在 default 階段是 `random`，之後是 `min_w`（見第 4 節）。
3. 兩次的梯度累加後，做一次 optimizer step（`train.py:148`）。

## 3. 增強哪些維度

- 【論文】增強寬度，也增強 expand ratio（論文 p.4 Figure 2 右圖）。**不增強深度**。
- 【論文】理由：作者說在 GPU 上，加寬的訓練時間開銷比加深小（論文 p.4 lines 220–222）。論文沒有給出這個比較的數據。
- 【論文】兩個超參數：
  - 增強倍率 `r`：最大寬度是 `r × w`。論文用同一個 `r` 控制所有運算（論文 p.4 lines 223–225）。
  - 多樣性因子 `s`（diversity factor）：控制增強網路的設定數。寬度在 `w` 到 `r×w` 之間等距分布（論文 p.4 lines 226–229）。
  - 論文範例：`r=3, s=2` → `[w, 2w, 3w]`。論文沒有定義 `s` 是否包含 `w`。「`s` 不含 `w`」是從這個例子推得的。
  - 論文同一段又寫「不同層可以用不同倍數」（論文 p.4 line 230）。兩句並存，論文沒有說明。官方實作的做法是：同一份寬度倍數清單，各層各自取樣。
  - 論文沒有寫實驗用的 `r` 與 `s` 值。
- 【官方】兩組清單（MobileNet 類模型）：
  - `aug_width_mult_list`：相對於基本寬度的寬度倍數清單，套在所有層的輸出通道（`models/netaug/mbv2.py:37-38, 71-73`）。
  - `aug_expand_list`：乘數，乘在每個 block 原本的 expand ratio 上（`models/netaug/mbv2.py:75-77`）。基本 expand ratio 是 `mid / in`（`models/base/layers.py:220-221`）。它不是絕對值。
  - 中間通道數 = expand ratio × 啟用的輸入通道數（`models/netaug/mbv2.py:201`）。
  - 不增強：kernel size、stride、分類器的輸出類別數（`models/netaug/mbv2.py:74, 78, 127`）。
- 【官方】`tinymbv2.sh`：expand `[1.0, 1.6, 2.2, 2.8]`，寬度倍數 `[1.0, 1.8, 2.6]`，影像大小 144，16 GPU（`bash/imagenet/tinymbv2.sh:1-5`）。因此寬度的 `r = 2.6`，expand 最大為基本值的 2.8 倍。
- 【官方】其他模型的腳本不同：
  - MBv3（`bash/imagenet/mbv3-0.35.sh:5-6`）：weight decay 3e-5、`base_lr` 0.1、寬度倍數 `[1.0, 2.0, 3.0]`。`base_lr × 16 GPU = 1.6`，是論文 0.4 的四倍。weight decay 也和論文的 4e-5 不同。論文沒有寫 MBv3 的設定。
  - MCUNet（`bash/imagenet/mcunet.sh:1-6`）：32 GPU、每 GPU batch 64、`base_lr` 0.0125、影像 176、寬度倍數 `[1.0, 1.6, 2.2]`。`0.0125 × 32 = 0.4`，和論文一致。
  - ProxylessNAS（`bash/imagenet/proxylessnas-0.35.sh:5`）：寬度倍數 `[1.0, 1.8, 2.6]`。
- 【官方】寬度取樣的粒度（MobileNetV2 系列，`models/netaug/mbv2.py:179-191`）：
  - 只有 `block.shortcut is None` 的 block 取樣輸出寬度。在 MobileNetV2 系列的基礎模型，這就是每個 stage 的第一個 block（`models/base/mbv2.py:74-76`）。
  - 其他 block（殘差連接）沿用前一個 block 的輸出寬度（`mbv2.py:190-191`）。程式沒有寫理由。「殘差相加需要通道數一致」是從結構推得的。
  - stem 的兩層（`mbv2.py:152-166`）與 head 的第一個 block、head 的 1×1 conv（`mbv2.py:212-225`）也取樣。
  - expand ratio 每個 block 獨立取樣，包括殘差連接的 block（`mbv2.py:192-200`）。
  - MBv3 的 stem 點卷積寬度直接等於 stem conv 的寬度，不另外取樣。MBv3 的 head 取樣 `op_list[0]` 與 `op_list[2]`（`models/netaug/mbv3.py:172-185, 231-253`）。
- 【論文】論文沒有寫寬度取樣的粒度。

## 4. 訓練排程（官方實作）【官方】

設 `remaining = n_epochs + warmup - epoch`（`train.py:244-248`）。條件依序判斷（`train.py:250-259`）：

| 條件 | 模式 | 意義 |
|---|---|---|
| 預設（`remaining` 大於 `stop_aug_w_epoch`） | `random` | 寬度與 expand 都隨機取樣 |
| `remaining <= stop_aug_w_epoch` | `min_w` | 寬度固定為最小，expand 仍取樣 |
| `remaining <= stop_aug_e_epoch`，且不符合 `min_w` 時才判 | `min_e` | expand 固定為最小，寬度仍取樣 |
| `remaining <= stop_netaug_epoch` | 關閉 | 只訓練小模型 |

- 條件是依序判斷的。`min_w` 先判。`min_e` 只在 `min_w` 不成立時才會判到。
- `configs/netaug.yaml` 預設：`stop_aug_w_epoch: 75`、`stop_aug_e_epoch: 0`、`stop_netaug_epoch: 0`、150 epoch、5 warmup epoch（`configs/netaug.yaml:13, 16, 23-25`）。
- README 說寬度增強在訓練後半段停止（`README.md:72`）。
- **已解決**：`tinymbv2.sh` 沒有傳入 `stop_aug_*`。`partial_update_config` 是遞迴合併（`utils/misc.py:82-91`），所以用 yaml 預設值（75 / 0 / 0）。
- tinymbv2 的實際排程（共 155 epoch，含 warmup 5）：
  - 第 0 到第 79 個 epoch（0-indexed，共 80 個，含 warmup 的 5 個）為 `default`（random）。
  - 第 80 到第 154 個 epoch（共 75 個）為 `min_w`。
  - 不會進入 `min_e`，也不會關閉增強。`remaining` 的最小值是 1，大於 `stop_aug_e_epoch` 與 `stop_netaug_epoch`（都是 0）。
  - 紀錄檔的 epoch 編號是 `epoch + 1 − warmup`。所以 `default` 紀錄到 75，`min_w` 紀錄從 76 到 150（`train.py:282`）。
  - 增強從第一個 epoch 就開始，包括 warmup（`train.py:250-252`）。

其他訓練設定：

- 【論文】batch 2048（16 GPU）、Nesterov SGD 動量（momentum）0.9、weight decay 4e-5、學習率 0.4 餘弦衰減到 0（論文 p.5 lines 269–272）。論文沒有寫 warmup。
- 【論文】ImageNet 預設 150 epoch；ImageNet-21K-P 預設 20 epoch；label smoothing 0.1 只用在 ImageNet（論文 p.5 lines 271–273）。
- 【官方】每 GPU 的 batch 是 128（`configs/netaug.yaml:4`），16 GPU 合計 2048。
- 【官方】學習率 = `base_lr × GPU 數`，即 `0.025 × 16 = 0.4`（`train.py:200`；`configs/netaug.yaml:15`）。
- 【官方】學習率排程：warmup 5 epoch，學習率從 0.025 線性升到 0.4，每個 iteration 更新。之後餘弦衰減到 0，餘弦長度是 150 epoch（`utils/lr_scheduler.py:24-38`；`train.py:205-210, 149`）。
- 【官方】動量 0.9 與 `nesterov=True` 寫死在 `train.py:198-203`，不在 yaml 裡。
- 【官方】名稱含 `bias` 或 `norm` 的參數不做 weight decay（`train.py:183-197`）。BN 模組的屬性名是 `norm`，所以 BN 參數符合條件（`models/netaug/layers.py:402`）。
- 【官方】label smoothing 0.1 同時用在小模型與增強網路的 loss（`train.py:131, 141, 212-214`）。驗證用一般的 cross entropy（`train.py:55, 72`）。
- 【官方】`netaug.yaml` 的 `dropout_rate` 是 0（`configs/netaug.yaml:10`）。
- 【官方】訓練紀錄的 `train_top1` 與 `train_loss` 只來自小模型那次 forward（`train.py:131-133, 151-152`）。

## 5. 官方實作的技巧【官方】

### 5.1 通道排序

- 只在 `default` 階段的 epoch 開始時執行（`sort_channel` 為 true 時，`train.py:260-263`）。tinymbv2 是第 0 到第 79 個 epoch，共 80 次。`min_w` 階段不排序。
- 重要度 = 讀取者權重的 L1 範數。具體做法是 pointwise conv 權重絕對值的總和，對它的**輸入**維度求和，由大到小排序（`models/netaug/utils.py:171-177`）。
- 同一個排列套用到相連的層：pointwise conv 的輸入、depthwise conv 與其 BN、inverted conv 與其 BN、SE 層（`models/netaug/utils.py:179-185`；SE 在 `utils.py:154-168`）。
- 排的是 inverted block 的**中間（expand）通道**。block 的輸出通道與殘差連接不排序。
- stem 的兩層與 head 的 1×1 conv 不排序。MBv2 系列的 `all_blocks` 包含 head 的第一個 block；MBv3 的 `all_blocks` 不包含 head（`models/netaug/mbv2.py:141-148, 275-277`；`models/netaug/mbv3.py:162-168`）。
- 排序的方法定義在 `NetAugMobileNetV2`。MBv3、MCUNet、ProxylessNAS、tinymbv2 都繼承它。
- 目的：讓小模型取到的前段通道是最重要的通道（`README.md:71`）。
- 【推論】排序直接改寫權重（`param.data`，`utils.py:139-141`），但 SGD 的動量緩衝（momentum buffer）沒有跟著重排（`train.py:198-203, 262`）。這會讓動量對到錯的通道。影響沒有實驗。
- 【論文】論文沒有寫通道排序。

### 5.2 BN 處理

- `DynamicBatchNorm2d` 依啟用寬度切片 running 統計值與 affine 參數（`models/netaug/layers.py:286-308`）。
- **增強前向也會更新 running 統計值**。小模型與增強網路的前向都在 train mode，各自更新啟用的通道切片，動量為 0.1（`layers.py:320-355`）。
- 驗證前，先 `set_active("min")`，再用 sub_train 做重新校正 BN 統計值（`reset_bn`）。多 GPU 時先跨 GPU 平均 batch 統計值（`train.py:276-278, 47-53`；`models/netaug/utils.py:33-55`）。
  - 條件：config 有 `netaug` 區段（`use_netaug`，`train.py:278, 386`）。與 mode 無關。沒有 netaug 的 baseline 不做。
  - 每個 epoch 都做，在驗證前。
  - `sync=True` 寫死在 `train.py:52`。
  - 只改寫啟用的前段通道（`models/netaug/utils.py:112-117`）。
- sub_train 的內容：
  - 從訓練集固定抽取最多 16,000 張（seed `937162211`，`setup.py:86-94`）。
  - 用的是訓練集的隨機增強 transform（RandomResizedCrop、flip、ColorJitter，`setup.py:57-70`）。
  - 在建立 loader 時就預先抓成 list（`setup.py:155`）。所以每個 epoch 用的是同一批已增強的影像。
  - 16 GPU 時，每個 rank 用 DistributedSampler 分到 1,000 張（`setup.py:133-142`）。batch 128 且 `drop_last=True`，每個 rank 只用 7 個 batch，共 896 張。16 GPU 合計 **14,336 張，不是 16,000 張**。這是程式推論，沒有實際執行。
- 「變異數」的算法：`sync=True` 時，先算全域 batch 的平均，再算變異數，最後平均各 batch 的結果（`models/netaug/utils.py:45-55, 72-73`）。這不是整體資料的變異數，而是各 batch 變異數的平均。這是程式行為。
- 驗證後存的 `best.pt` 內含該 epoch 重新校正後的 BN 統計值（`train.py:278-308`）。
- 【論文】論文沒有寫 BN 重新校正。README 說 NetAug 的 BN 統計值不準，所以要重新校正（`README.md:73`）。

### 5.3 初始化

- 用 `kaiming_uniform`，不用 He fan-out。README 說 fan-out 在大 expand ratio 時效果差。README 把 `kaiming_uniform` 連到 PyTorch 的 conv 初始化（`README.md:70`；`configs/netaug.yaml:18`；`utils/init.py:29-30`）。
- Linear 權重用 trunc normal（std 0.02）（`utils/init.py:38-39`）。BN 的 gamma 設為 1，bias 設為 0（`utils/init.py:35-37`）。
- **殘差連接最後一個 BN 的 gamma 沒有設為 0。** 所有 BN 的 gamma 都是 1。grep 找不到其他 zero-init（`utils/init.py:35-37`；其他 `zero_` 只用在 conv 與 Linear 的 bias）。
- 初始化套在**最大的增強網路**上，不是小模型（`train.py:386-401`）。
- 官方 repo 的 `configs/default.yaml` 用 `he_fout`，但它只在 `bash/transfer/*.sh` 出現（fine-tune）。ImageNet baseline 的設定不在 repo 內。
- 【論文】論文沒有寫 baseline 用哪一種初始化。

### 5.4 匯出

- 載入 `best.pt` → `set_active("min")` → 跑一次假輸入 → `export()` → 存成 `target.pt`（`train.py:310-331`）。只在 `use_netaug` 時做，只有 master 寫檔。
- `best.pt` 是依驗證 top-1 選出的最佳 epoch，不是最後一個 epoch（`train.py:279-280, 303-308`）。
- `export()` 建出一般的 `MobileNetV2` 類別。動態層用 `active_state_dict()` 載入切片後的權重（`models/netaug/mbv2.py:227-273`；`models/netaug/layers.py:118-139`）。
- 匯出的 BN 統計值來自 best epoch 的重新校正。

### 5.5 動態層的啟用寬度

- 動態層從輸入張量推得輸入通道數（`x.shape[1]`），只需要設定輸出寬度（`models/netaug/layers.py:103, 244, 313`）。
- 權重切片方式：一般層是 `weight[:out, :in]`（`layers.py:89-94`）；depthwise 層是 `weight[:in, :in]`（`layers.py:170-175`）。

## 6. 實驗結果（已對照 PDF）

- 【論文】ImageNet，MobileNetV2-Tiny（r144）：150 epoch 時 51.7% → 53.3%（+1.6%，Table 1，p.6）。
- 【論文】300 epoch：52.4% → 53.7%（+1.3%，Table 3，p.7）。訓練成本 420 → 490 GPU 小時，多 16.7%。§3.3 寫 150 epoch 的成本是 210 → 245 GPU 小時（p.5 lines 252–254）。兩者比例都是 16.7%。
- 【論文】Table 3（p.7，300 epoch）：Mixup、AutoAugment、RandAugment、DropBlock 都讓 MobileNetV2-Tiny 變差（−0.7% 到 −3.7%）。Dropout 兩列（kp=0.9 −0.7%，kp=0.8 −1.4%）的基準是 150 epoch（51.7%），不是 300 epoch。
- 【論文】Table 1 最大增益：MobileNetV3 w0.35 r160，58.1% → 60.3%（+2.2%）。
- 【論文】Figure 3（p.6）：達到相同準確率，ImageNet 少用 75% epoch，ImageNet-21K-P 少用 83% epoch。
- 【論文】偵測（YoloV3 + 小 backbone）：
  - 在相近 AP50 下，Pascal VOC 少 41% MACs，COCO 少 38% MACs（摘要；論文 p.9 lines 779–783；Figure 5）。
  - Table 4（遷移學習，偵測輸入 416）：MobileNetV3 w0.35 的 AP50，VOC 從 63.6 → 65.3（+1.7），COCO 從 29.2 → 30.7（+1.5）。
  - 引言寫 VOC +3.36%、COCO +1.8%（論文 p.2 line 104）。這兩個數字來自 Figure 5 的圖上數值：MobileNetV3-VOC 在 87.8M MACs 時 56.96 對 53.6（+3.36）；MobileNetV3-COCO 在 290.3M MACs 時 30.2 對 28.4（+1.8）。
  - 單位不一致：引言寫 AP50，Figure 5 的標示寫 mAP（VOC）。引用時請註明採用哪一處。
- 【論文】**大模型會變差**：ResNet50 從 76.8% 降到 76.5%（Table 1，p.6）。
  - Figure 4（p.8）：ResNet50 的訓練準確率上升、驗證準確率下降。作者解讀為過擬合會更嚴重（論文 p.7 lines 409–416）。
  - NetAug 針對的是欠擬合的小模型。

## 7. 官方 repo 結構

```text
netaug/
├── train.py, eval.py, setup.py
├── configs/default.yaml, configs/netaug.yaml
├── bash/imagenet/*.sh（mbv2-0.35, mbv3-0.35, mcunet, proxylessnas-0.35, tinymbv2）
├── bash/transfer/*.sh（car, cub200, flowers, food, pets）
├── bash/eval.sh
├── models/base/        一般模型（mbv2, mbv3, mcunet, proxylessnas, tiny_mbv2；layers.py）
├── models/netaug/
│   ├── layers.py       DynamicModule, DynamicConv2d, DynamicDepthwiseConv2d, DynamicLinear,
│   │                   DynamicBatchNorm2d, DynamicConvLayer, DynamicDepthwiseConvLayer,
│   │                   DynamicLinearLayer, DynamicDsConvLayer, DynamicSE,
│   │                   DynamicInvertedBlock, DynamicSeInvertedBlock
│   ├── mbv2.py         NetAugMobileNetV2（set_active, sort_channels, export）
│   ├── mbv3.py         NetAugMobileNetV3（all_blocks 不含 head）
│   ├── mcunet.py       NetAugMCUNet
│   ├── proxylessnas.py NetAugProxylessNASMobile（繼承 NetAugMobileNetV2）
│   ├── tiny_mbv2.py    NetAugTinyMobileNetV2
│   └── utils.py        reset_bn, aug_width, sync_width, sort_param, sort_norm, sort_se, sort_channels_inner
└── utils/              criterion, distributed, init, lr_scheduler, metric, misc, profile
```

## 8. 論文與官方實作的對照

本表已逐項對照官方實作。

| 項目 | 論文 / 官方實作 |
|---|---|
| 任務與模型 | 論文：ImageNet 分類（MobileNet 類）；偵測用 YOLOv3 風格頭。官方：同上，另有 MCUNet、ProxylessNAS、MBv3 的腳本 |
| 增強維度 | 寬度 + expand ratio，不增強深度（論文 p.4 Figure 2） |
| 寬度倍數清單 | 論文：等距（linspace，p.4 lines 228–229）。官方：相對於基本寬度的固定倍數清單，如 `[1.0, 1.8, 2.6]` |
| 取樣粒度 | 論文：未寫。官方：`shortcut is None` 的 block 取樣輸出寬度（MBv2 系列即每個 stage 第一個 block）；殘差連接沿用；stem 與 head 也取樣；expand 每個 block 取樣 |
| 每步增強網路數 | 1 個 |
| Loss | 論文：`L(base) + 1.0 × L(aug)`，兩次 backward 累加梯度。官方：同左，兩者都用 label smoothing |
| KD | 論文：NetAug 本身不含 KD；Table 2 測試 NetAug + KD 組合（教師 Assemble-ResNet50）。官方實作：無 KD |
| 預訓練最大網路 | 無 |
| 停止增強 | 論文：未寫。官方：依剩餘 epoch，`stop_aug_w_epoch` 等（tinymbv2 從第 0-indexed 第 80 個 epoch 起停止寬度增強，見第 4 節） |
| 通道排序 | 論文：未寫。官方：只在 default 階段每個 epoch 開始時，依 pointwise 權重的 L1 排序 inverted block 的中間通道；同一排列套到相連層 |
| 重新校正 BN 統計值 | 論文：未寫。官方：每個 epoch 驗證前，用最多 16,000 張固定子集（含訓練增強） |
| 啟用輸入通道 | 論文：未寫。官方：從 `x.shape[1]` 推得 |
| 分組卷積 | 官方：有 `DynamicDepthwiseConv2d`（論文未寫） |
| 匯出 | 官方：`set_active("min")` → `export()` |

## 9. 論文的補充內容（上一版沒有寫）

### 9.1 論文的訓練與遷移設定

- 【論文】細粒度分類（Food101、Flowers102、Cars、Cub200、Pets）：用 ImageNet 預訓練權重初始化，batch 256，4 GPU，微調 50 epoch（論文 p.5 lines 274–276）。
- 【論文】Pascal VOC 偵測：VOC 2007 trainval 加 VOC 2012 trainval 訓練，VOC 2007 test 測試；200 epoch，batch 64，8 GPU（論文 p.5 lines 277–279）。
- 【論文】COCO 偵測：COCO2017 train 訓練，COCO2017 val 測試；120 epoch，batch 128，16 GPU（論文 p.5 lines 279–281）。
- 【論文】偵測架構：YoloV3，backbone 換成小網路，偵測頭的一般卷積換成 depthwise 卷積。backbone 用 ImageNet 預訓練權重，偵測頭隨機初始化（論文 p.5 lines 282–285）。
- 【論文】論文沒有寫各任務的 dropout 與學習率。官方 transfer 腳本各模型的 dropout 與 `base_lr` 不同（`bash/transfer/car.sh:1-39`，50 epoch，batch 64 × 4 GPU）。
- 【官方】訓練資料增強（`setup.py:57-70`）：RandomResizedCrop、RandomHorizontalFlip、ColorJitter（brightness 32/255、saturation 0.5）、ToTensor、Normalize（ImageNet 平均與標準差）。驗證用 Resize 到 `ceil(image_size / 0.875)` 再 CenterCrop（`setup.py:72-84`）。論文沒有寫增強的細節。

### 9.2 補充的表與圖

- 【論文】Table 1 其他模型（論文 p.6）：ProxylessNAS-Mobile +1.7%；MCUNet +1.2%；MobileNetV3 +2.2%。NAS 設計的模型也有增益。
- 【論文】Table 1 的 MobileNetV2 系列（論文 p.6 lines 348–351）：寬度越小，增益越大（w1.0 +0.9%，w0.35 +1.5%）。作者猜測小模型容量低，欠擬合更嚴重。
- 【論文】Table 2（論文 p.7）：與知識蒸餾（KD）比較。所有模型的 KD 教師都是 Assemble-ResNet50（論文 p.6 lines 359–362）。

  | 模型 | Baseline | KD | NetAug | NetAug + KD |
  |---|---|---|---|---|
  | MobileNetV2-Tiny | 51.7% | 52.8% | 53.3% | 54.4% |
  | MobileNetV2 w0.35 r160 | 56.3% | 57.0% | 57.8% | 59.2% |
  | MobileNetV3 w0.35 r160 | 58.1% | 59.8% | 60.3% | 61.5% |
  | ProxylessNAS w0.35 r160 | 59.1% | 60.4% | 60.8% | 61.5% |

  NetAug 的增益比 KD 多 +0.4% 到 +0.8%（依模型不同，論文 p.6 lines 363–366）。兩者合併後，增益再加：MobileNetV2-Tiny 達到 +2.7%。作者說兩者的機制不同，互補（論文 p.2 §2）。

- 【論文】Figure 3 的數字（論文 p.6）：

  | 資料集 | epoch | NetAug | Baseline |
  |---|---|---|---|
  | ImageNet | 150 | 53.3% | 51.7% |
  | ImageNet | 300 | 53.7% | 52.4% |
  | ImageNet | 600 | 54.4% | 53.1% |
  | ImageNet-21K-P | 20 | 24.2% | 22.4% |
  | ImageNet-21K-P | 60 | 25.0% | 23.4% |
  | ImageNet-21K-P | 120 | 25.4% | 24.1% |

- 【論文】Table 4（論文 p.8）：遷移學習，權重由不同方法預訓練。欄位依序為 ImageNet、Food101、Flowers102、Cars、Cub200、Pets、VOC AP50、COCO AP50。MobileNetV2 w0.35 r160 的列：

  | 預訓練方法 | ImageNet | Food101 | Flowers102 | Cars | Cub200 | Pets | VOC | COCO |
  |---|---|---|---|---|---|---|---|---|
  | Baseline (150) | 56.3 | 76.4 | 90.8 | 76.9 | 69.0 | 81.9 | 60.4 | 24.7 |
  | Baseline (300) | 57.0 | 76.5 | 90.3 | 75.8 | 69.6 | 81.7 | 60.8 | — |
  | Baseline (600) | 57.5 | 76.8 | 89.7 | 74.7 | 69.5 | 81.7 | 61.3 | — |
  | KD | 57.0 | 76.4 | 91.6 | 78.4 | 68.4 | 80.3 | 60.0 | 24.3 |
  | NetAug | 57.8 | 77.4 | 92.4 | 79.8 | 68.8 | 82.3 | 62.4 | 25.4 |

  MobileNetV3 w0.35 r160 的列：Baseline (150) 58.1 / 76.6 / 92.1 / 75.8 / 69.3 / 83.1 / 63.6 / 29.2；KD 59.8 / 76.9 / 92.6 / 77.2 / 69.2 / 82.9 / 63.4 / 28.9；NetAug 60.3 / 78.3 / 93.0 / 78.9 / 70.4 / 84.6 / 65.3 / 30.7。

  作者的結論（論文 p.8 lines 659–669）：
  - ImageNet 準確率高，不代表遷移學習好。
  - KD 預訓練的權重傷害 Cub200、Pets 與所有偵測任務（以上表的 MobileNetV2 KD 列為例，四項都低於 baseline）。
  - 訓練更多 epoch 也會傷害細粒度分類：MobileNetV2 的 Baseline (300)、(600) 在 Flowers102、Cars、Pets 三項低於 Baseline (150)。
  - NetAug 預訓練的權重在多數任務最好。

- 【論文】Table 5（論文 p.9）：凍結權重的 tiny transfer（只訓練 bias 與 lite residual 模組，Cai et al. 2020b）。NetAug 在兩個模型的五個細粒度資料集都較好（欄位：Food101、Flowers102、Cars、Cub200、Pets）：

  | 模型 | 方法 | Food101 | Flowers102 | Cars | Cub200 | Pets |
  |---|---|---|---|---|---|---|
  | MobileNetV2 w0.35 r160 | Baseline | 67.33 | 89.04 | 58.28 | 57.75 | 78.88 |
  | MobileNetV2 w0.35 r160 | NetAug | 68.67 | 90.40 | 60.02 | 58.39 | 79.37 |
  | MobileNetV3 w0.35 r160 | Baseline | 67.53 | 89.19 | 51.18 | 57.99 | 80.32 |
  | MobileNetV3 w0.35 r160 | NetAug | 69.74 | 90.73 | 56.21 | 58.94 | 82.04 |

- 【論文】Figure 5 的偵測結果（論文 p.9）：偵測 mixup（Zhang 2019b）對小模型變差，尤其是 YoloV3 + MobileNetV2 w0.35（論文 p.9 lines 785–788）。作者說這表示小模型的欠擬合問題也存在於影像分類以外的任務。
- 【論文】Appendix A，Table 6（論文 p.13）：正則化強度的消融（300 epoch）。
  - Mixup α=0.1：−0.2%；α=0.2：−0.7%。
  - DropBlock kp=0.95：−1.8%；kp=0.9：−3.7%。
  - RandAugment N=1, M=9：−0.9%；N=2, M=9：−2.8%。
  - NetAug：+1.3%。
  - 結論：正則化越強，準確率損失越大。
- 【論文】Appendix B，Table 7（論文 p.13）：替代的訓練設定（150 epoch，baseline 51.7%）。
  - weight decay 1e-5：51.6%；1e-6：51.3%。
  - 弱資料增強（無增強）：50.8%。
  - batch 1024：51.5%；batch 256：51.7%。
  - NetAug：53.3%。
  - 作者說這些替代方法在 ImageNet 上都不如 NetAug。
- 【論文】論文沒有報告 seeds、重複次數或標準差。論文沒有 limitations 一節。
- 【論文】論文的 footnote 1（論文 p.5）給出程式與預訓練權重的連結。

### 9.3 機制：論文只有直覺，沒有推導

- 【論文】論文給的機制只有一句直覺：小模型的容量有限，比大模型更容易卡在局部極小值（local minimum），因此訓練與測試表現都較差（論文 p.3 lines 159–161）。
- 【論文】論文**沒有推導**。全文沒有理論分析或證明。grep 找不到 theorem、proof、derive、convergence。
- 【論文】論文用 Figure 4 的學習曲線當作間接證據。它支持「欠擬合」的解讀，但沒有直接檢驗「卡在局部極小值」的假說。
- 【論文】論文說增強網路提供「額外的監督」（論文 p.2 §1；p.4 Figure 2 caption），但沒有說明為什麼這份額外監督會讓小模型變好。

### 9.4 Figure 4 的用途：適用性測試

- 【論文】Figure 4（論文 p.8）比較有無 NetAug 的學習曲線（論文 p.7 lines 409–416）。
  - MobileNetV2-Tiny：訓練與驗證準確率都上升。作者解讀為 NetAug 減少了欠擬合。
  - ResNet50：訓練準確率上升，驗證準確率下降。作者解讀為過擬合加重。
- 【論文】作者用這個對照來判斷 NetAug 適用於哪一類模型：只適用於欠擬合的小模型。

### 9.5 只有官方實作的細節

- 【官方】排序、重新校正 BN 統計值、warmup、停止增強的排程、初始化細節、匯出，論文都沒有寫；只來自官方實作。
- 【官方】重新校正 BN 統計值使用的 sub_train 是增強後的影像，並且固定不變（第 5.2 節）。
- 【官方】`eval.py` 的 `--reset_bn` 預設不做（`eval.py:49, 80`）。

### 9.6 漏讀檢查（第三步）

以下項目在上一版與本次抽取單都沒有寫，本次補上：

- 【論文】第 9.1 節的細粒度、VOC、COCO 設定（論文 p.5 lines 274–285）。
- 【官方】訓練資料增強與驗證資料前處理（`setup.py:57-84`）。
- 【官方】`netaug.yaml` 的 `dropout_rate: 0`（`configs/netaug.yaml:10`）。
- 【官方】`sync` 與 `generator`：`netaug.yaml:27` 的 `sync: false`。每次 `set_active` 的取樣都用 `exp_config["generator"]`（`train.py:135-139`）。這個 generator 在每個 rank 上都用同一個 seed（`train.py:414-415`），所以所有 rank 會取樣到同一組增強網路（程式推論，沒有實際執行）。
- 【官方】`--resume` 會把 seed 改成 `int(time.time())`（`train.py:352-353`）。checkpoint 沒有存 generator 的狀態（`train.py:290-296`），所以續訓後的取樣序列與原本不同。
- 【官方】最佳模型依驗證 top-1 選出（`train.py:279-280, 303-308`）。
- 【官方】`reset_bn` 實際用的影像數是 14,336 張（16 GPU），不是 16,000 張（見第 5.2 節）。
- 【論文】論文 footnote 1 與引用格式（論文 p.5；README）。
