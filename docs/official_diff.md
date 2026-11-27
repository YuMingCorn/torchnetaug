# torchnetaug 與官方實作的差異

**繁體中文** | [English](official_diff.en.md)

本文件逐項對照 torchnetaug 與官方實作的通道排序與訓練迴圈。官方實作指 tinyml 的 NetAug 實作；深度增強另參考 Once-for-All。設計見 [design.md](design.md)，術語見 [glossary.md](glossary.md)。

## 重點

1. **排序公式相同**：兩者都用讀取者權重的 L1 遞減排序。只有一個 dense 讀取者時，重要度逐元素相等。
2. **範圍不同**：官方只排序 `inner` 群組（expand 與 SE 隱藏通道）。本套件預設排序所有可增強群組，`scope="inner"` 對應官方的範圍。
3. **修正一個偏離**：官方排序時不重排 optimizer 狀態。本套件會重排，所以動量和參數的通道一致。
4. **範例的兩個差異**：範例跳過第 0 個 epoch 的排序，也不用 label smoothing。
5. **連結都釘在固定 commit**：每列的兩格都連到帶行號的原始碼。

## 「性質」欄的值

| 值 | 意思 |
|---|---|
| 等價 | 結果相同，寫法可以不同 |
| 刻意擴充 | 官方沒有，或範圍較小；本套件刻意增加 |
| 偏離-修正 | 和官方不同，因為官方的做法會出錯 |
| 偏離-未說明 | 和官方不同，以前沒有文件說明；差異現在寫在表下 |
| 待核驗 | 還沒有實測 |

## 差異表

行號連結：官方是 tinyml `0e58f314dfe12967d6165f45238b1ea77b0f7a14`，torchnetaug 是 `13a9f0e562ab34836a8d449b4f4ad6a85f0e655d`。

| 項目 | 官方實作 | torchnetaug | 性質 |
|---|---|---|---|
| 1. 重要度公式 | 一個讀取者（block 的 point conv）的 L1：`torch.sum(torch.abs(w), dim=(0, 2, 3))`，遞減排序。[`utils.py` L174–L177](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L174-L177) | 加總所有 dense 讀取者在對應段上的 L1，stable 遞減排序。[`netaug.py` L662–L699](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L662-L699)、[`netaug.py` L626–L648](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L626-L648) | 刻意擴充（只有一個 dense 讀取者時等價） |
| 2. 備用算法（沒有讀取者） | 沒有。官方支援的 block 一定有 point conv 讀取者；其他 block 直接報錯。[`utils.py` L186–L187](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L186-L187) | 改用 dense 生產者輸出權重（dim 0）的 L1。[`netaug.py` L639–L647](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L639-L647) | 刻意擴充 |
| 3. 範圍：inverted block 的 expand 通道（`inner`） | 排序。[`utils.py` L171–L185](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L171-L185)、[`mbv2.py` L275–L277](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/mbv2.py#L275-L277) | 排序（`scope="all"` 與 `scope="inner"` 都含）。[`netaug.py` L763–L776](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L763-L776)、[`netaug.py` L813–L818](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L813-L818) | 等價（範圍相同；帶 SE 的 block 的重要度見第 1 列） |
| 4. 範圍：SE 隱藏通道（`inner`） | 排序，讀取者是 SE 的 expand conv。[`utils.py` L154–L168](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L154-L168) | 排序。群組分類把它判為 `inner`。[`graph.py` L152–L157](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/graph.py#L152-L157) | 等價 |
| 5. 範圍：stem 輸出（`closed`） | 增強寬度（[`mbv2.py` L33–L58](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/mbv2.py#L33-L58)），不排序：`all_blocks` 不含 stem（[`mbv2.py` L141–L148](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/mbv2.py#L141-L148)） | 預設 `scope="all"` 會排序；`scope="closed"` 也會。[`netaug.py` L763–L770](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L763-L770) | 刻意擴充 |
| 6. 範圍：stage 主幹（`residual`） | 增強寬度（[`mbv2.py` L71–L73](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/mbv2.py#L71-L73)），不排序：point conv 只重排 dim 1（expand 通道），不重排 dim 0（[`utils.py` L179–L179](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L179-L179)） | 預設 `scope="all"` 會排序；所有生產者的 dim 0 用同一個排列。[`netaug.py` L813–L819](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L813-L819)、[`netaug.py` L607–L611](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L607-L611) | 刻意擴充 |
| 7. 範圍：head 的 final conv（`closed`） | 增強寬度（[`mbv2.py` L112–L121](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/mbv2.py#L112-L121)），不排序（[`mbv2.py` L141–L148](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/mbv2.py#L141-L148)） | 預設 `scope="all"` 會排序。[`netaug.py` L813–L819](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L813-L819) | 刻意擴充 |
| 8. 範圍：`concat-segment` 與其他 block 類型 | 不支援，`raise NotImplementedError`。[`utils.py` L186–L187](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L186-L187) | 支援：每段依自己的偏移量排列。[`netaug.py` L612–L623](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L612-L623) | 刻意擴充 |
| 9. 套用排列的張量 | point conv `weight` 的 dim 1；depthwise 與 expand conv 的 `weight` dim 0 與 BN；SE。[`utils.py` L178–L185](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L178-L185)；BN 含 running 統計值：[`utils.py` L144–L151](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L144-L151) | 生產者的 `weight`、`bias`（dim 0）；member 層的參數與 running 統計值（dim 0）；讀取者的 `weight`（dim 1）。[`netaug.py` L602–L624](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L602-L624) | 等價 |
| 10. 原地修改 | `param.data.copy_(index_select(...))`，保留同一個 Parameter 物件。[`utils.py` L134–L141](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L134-L141) | `index_select` 後 `index_copy_`，保留同一個 Parameter 物件。[`netaug.py` L889–L892](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L889-L892) | 等價 |
| 11. optimizer 狀態 | 不處理。optimizer 在訓練前建立一次（[`train.py` L198–L203](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L198-L203)），每個 epoch 排序（[`train.py` L260–L263](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L260-L263)），排序不讀寫動量 | 傳入 `optimizer` 時，形狀相同的狀態張量用同一個排列。[`netaug.py` L886–L888](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L886-L888) | 偏離-修正 |
| 12. `.grad` | 不處理。排序在 epoch 開始、`zero_grad` 之前（[`train.py` L260–L263](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L260-L263)、[`train.py` L125–L125](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L125-L125)），這時的梯度不會被使用 | 參數有 `.grad` 時用同一個排列。[`netaug.py` L882–L885](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L882-L885) | 刻意擴充 |
| 13. 排序時機 | 只在 `default` 階段（寬度與 expand 都增強），每個 epoch 開始時，含第 0 個 epoch。[`train.py` L250–L263](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L250-L263) | 函式庫不規定時機。範例每 `--sort-every` 個 epoch 排序，跳過第 0 個 epoch，停止增強（`aug_until`）後不排序。[`cifar_netaug.py` L498–L499](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/examples/cifar_netaug.py#L498-L499)、[`cifar_netaug.py` L403–L403](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/examples/cifar_netaug.py#L403-L403) | 偏離-未說明（見表下說明） |
| 14. DDP | 每個 rank 各自呼叫 `model.module.sort_channels()`，不同步排列。[`train.py` L262–L262](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L262-L262) | 每個行程各自計算排列，不同步。`netaug_step` 會解開 `.module`（[`train.py` L45–L46](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/train.py#L45-L46)），但 DDP 未實際測試。`sort_channels` 沒有跨行程同步的程式（[`netaug.py` L809–L819](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L809-L819)） | 待核驗（推論：各 rank 權重相同時結果相同） |
| 15. 深度增強 | 沒有。`set_active` 只改寬度。[`mbv2.py` L150–L225](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/mbv2.py#L150-L225) | `DepthStack`（[`depth.py` L40–L56](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/depth.py#L40-L56)）。額外 block 的層不計入原模型就有的群組的重要度；只有整個在額外 block 內的群組才用它們排序（[`netaug.py` L626–L648](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/netaug.py#L626-L648)） | 刻意擴充 |
| 16. optimizer 設定 | SGD、momentum 0.9、nesterov，lr = `base_lr` × 行程數。[`train.py` L198–L203](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L198-L203)、[`netaug.yaml` L12–L18](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/configs/netaug.yaml#L12-L18) | 範例：SGD、momentum 0.9、nesterov，lr = `--lr`。[`cifar_netaug.py` L387–L392](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/examples/cifar_netaug.py#L387-L392) | 等價（lr 與 epoch 數不同，屬配方） |
| 17. weight decay 分組 | 名稱含 `bias` 或 `norm` 的參數不做 weight decay。[`train.py` L183–L197](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L183-L197) | 範例：`ndim <= 1` 的參數不做 weight decay。[`cifar_netaug.py` L385–L388](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/examples/cifar_netaug.py#L385-L388) | 等價（判定方式不同，結果相同：BN 參數與 bias） |
| 18. `zero_grad`、`backward`、`step` 的順序 | `zero_grad` → base `backward` → 增強 `backward` → `step` → 排程 `step`。[`train.py` L125–L149](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L125-L149) | `netaug_step` 做兩次 `backward`（[`train.py` L48–L59](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/train.py#L48-L59)）；範例接著 `step` → `zero_grad` → 排程 `step`（[`cifar_netaug.py` L507–L528](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/examples/cifar_netaug.py#L507-L528)） | 等價 |
| 19. loss | base loss 與增強 loss 的梯度直接相加（沒有係數）；label smoothing 0.1。[`train.py` L131–L142](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L131-L142)、[`criterion.py` L33–L40](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/utils/criterion.py#L33-L40) | 增強 loss 乘上 `alpha`（預設 1.0）（[`train.py` L18–L27](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/train.py#L18-L27)、[`train.py` L53–L56](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/src/torchnetaug/train.py#L53-L56)）；範例的 `crit` 沒有 label smoothing（[`cifar_netaug.py` L405–L406](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/examples/cifar_netaug.py#L405-L406)） | 偏離-未說明（見表下說明） |
| 20. lr 排程 | warmup 後 cosine，每個 iteration 更新。warmup 的起點 `warmup_lr` 就是 `base_lr`，而 optimizer 的 lr 是 `base_lr × 行程數`，所以單一行程時 warmup 期間 lr 固定為 `base_lr`，多行程時從 `base_lr` 升到 `行程數 × base_lr`。[`lr_scheduler.py` L24–L38](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/utils/lr_scheduler.py#L24-L38)、[`train.py` L198–L210](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L198-L210) | 範例：`lr_factor` 與 `LambdaLR`，從接近 0 線性 warmup 後 cosine，每個 iteration 更新。[`cifar_netaug.py` L397–L402](https://github.com/YuMingCorn/torchnetaug/blob/13a9f0e562ab34836a8d449b4f4ad6a85f0e655d/examples/cifar_netaug.py#L397-L402) | 偏離-未說明（配方：單一行程時官方實際上沒有 warmup） |

## 表下說明

- **排序時機（第 13 列）**：範例和官方的差異是範例跳過第 0 個 epoch。`--sort-every 1` 對應官方的「每個 epoch」。官方在寬度增強停止後不再排序；範例沒有分開的寬度與 expand 階段，所以在停止增強後不再排序。
- **loss（第 19 列）**：範例的 `crit` 不用 label smoothing，官方用 0.1。`alpha=1`（預設）時，兩次 `backward` 的梯度直接相加，和官方等價。
- **DDP（第 14 列）**：「各 rank 權重相同時結果相同」是推論，沒有實測。
- **optimizer 狀態（第 11 列）**：動量錯位的後果來自玩具實驗，數字見 [design.md 第 5.1 節](design.md#51-正確性證明)的定理 3。
- **lr 排程（第 20 列）**：官方的 `warmup_lr` 等於 `base_lr`，所以單一行程時 warmup 沒有作用；範例從接近 0 開始 warmup。這是配方差異，不影響套件本身。

## 範例：官方的範圍與公式

下面的程式檢查兩件事。第一，MobileNetV2 的 `inner` 群組只有一個 dense 讀取者（point conv），所以重要度等於官方的公式。第二，`scope="inner"` 只排序這些群組。

```python
import torch
import torchvision
from torchnetaug import NetAugModel

tiny = torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10)
model = NetAugModel(tiny, torch.randn(2, 3, 32, 32), width_mults=(1.0, 2.0))

kinds = model.group_kinds()
importance = model.channel_importance()
inner = {g for g, info in kinds.items() if info.kind == "inner"}
for g in inner:
    (reader,) = kinds[g].readers                       # 只有 point conv
    w = model.model.get_submodule(reader).weight
    assert torch.allclose(importance[g], torch.sum(torch.abs(w), dim=(0, 2, 3)))  # 官方的公式

perms = model.sort_channels(scope="inner")            # 官方的範圍
assert set(perms) == inner
```
