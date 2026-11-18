# torchnetaug 架構與通道排序

本文件說明 `torchnetaug` 的內部設計：它怎麼把任意 PyTorch 模型變成可切換寬度的模型，以及通道排序為什麼在殘差網路上也是正確的。使用方式請看 [README](../README.md)。

本文件中的 Python 範例都會由 `tests/test_docs.py` 實際執行，所以文件和程式碼不會脫節。

## 1. 總覽

NetAug 的目標是讓小模型（base）在訓練時，同時被嵌在更寬的模型裡一起訓練。所有寬度共用同一組權重，小模型使用前面的通道。訓練結束後只匯出小模型。

套件由四個檔案組成，資料流如下：

```text
原始模型 ──► graph.analyze() ──► ChannelGraph ──► NetAugModel ──► export() ──► 原始類別的小模型
             (torch.fx 追蹤)      (通道群組)        (最大寬度的動態模型)
                                                     │
                                  train.netaug_step ─┤  每一步：base loss + alpha × aug loss
                                  train.recalibrate_bn  評估或匯出前重新估計 BN 統計值
                                  NetAugModel.sort_channels  依重要度排序通道
```

| 檔案 | 內容 |
|---|---|
| `src/torchnetaug/graph.py` | `analyze()`、`ChannelGraph`、`LayerRole`、`TraceError`：找出通道群組 |
| `src/torchnetaug/layers.py` | `DynamicConv2d`、`DynamicLinear`、`DynamicBatchNorm`：以最大寬度存權重，執行時切片 |
| `src/torchnetaug/netaug.py` | `NetAugModel`：建立最大寬度模型、切換寬度、排序、匯出 |
| `src/torchnetaug/train.py` | `netaug_step()`、`recalibrate_bn()`：訓練輔助函式 |

## 2. 核心資料結構：通道群組（channel group）

**通道群組**是一組「寬度必須相同、通道順序也必須相同」的通道。例如：

- 一個 dense conv 的輸出通道，和所有讀取它的層的輸入通道。
- 殘差相加 `x + f(x)` 的兩邊。所以 ResNet 一個 stage 裡每個 block 的輸出，會併成**同一個**群組。
- batch norm 或 depthwise conv 的輸入與輸出。

每個群組有三個屬性：

- **大小**：原始模型中的通道數，也就是 base 寬度。
- **是否固定（fixed）**：固定的群組永遠維持原始寬度，也不排序。模型輸入、模型輸出，以及無法安全分析的地方都會被固定。
- **寬度選項**：`round(大小 × m)`，`m` 取自 `width_mults`。最大的選項就是最大寬度。

每個動態層在 `ChannelGraph.layers` 裡有一個 `LayerRole`：

| `kind` | 層的類型 | 意義 |
|---|---|---|
| `"dense"` | `nn.Conv2d`（`groups=1`）、`nn.Linear` | 輸出通道自成一個群組 `out_group`；輸入通道依序屬於 `in_groups` |
| `"member"` | `nn.BatchNorm*`、depthwise `nn.Conv2d` | 輸入通道等於輸出通道，都屬於 `in_groups` |

### 2.1 concat 與「段」（segment）

`torch.cat(..., dim=1)` 不會合併群組。它的輸出是一個**段列表**，每個被串接的群組佔一段。例如 DenseNet 式的串接：

```python
import torch
import torch.nn as nn
from torchnetaug import analyze


class DenseLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 6, 3, padding=1)
        self.c1 = nn.Conv2d(6, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(10)
        self.head = nn.Linear(10, 5)

    def forward(self, x):
        x0 = torch.relu(self.stem(x))
        x1 = torch.relu(self.c1(x0))
        y = self.bn(torch.cat([x0, x1], dim=1))  # 10 = 6 + 4 個通道，兩段
        return self.head(y.mean([2, 3]))


graph = analyze(DenseLike(), torch.randn(1, 3, 16, 16))
roles = graph.layers
stem_g, c1_g = roles["stem"].out_group, roles["c1"].out_group
assert roles["bn"].in_groups == (stem_g, c1_g)    # BN 讀兩段
assert roles["head"].in_groups == (stem_g, c1_g)  # Linear 也讀同樣兩段
print(graph.summary())
```

在最大寬度的權重裡，每一段有自己的固定位置（偏移量）。偏移量是前面各段的**最大**寬度總和。執行時每段只取前 `active` 個通道：

```text
head.weight 的輸入維度（最大寬度 = 18 + 12 = 30）
┌──────────── stem 段：offset 0，最大 18 ────────────┬────── c1 段：offset 18，最大 12 ──────┐
│ ■■■■■■ (base 6) □□□□□□□□□□□□                       │ ■■■■ (base 4) □□□□□□□□                │
└────────────────────────────────────────────────────┴──────────────────────────────────────┘
base 模式取的索引 = [0..5] ∪ [18..21]
```

### 2.2 白名單分析

`analyze()` 用 `torch.fx.symbolic_trace` 追蹤模型，再用 `ShapeProp` 算出每個節點的張量形狀，最後逐個節點套用規則。**只有白名單上的運算能讓通道穿透**，其他運算一律把它碰到的群組固定。這是保守的做法：不認得的運算只會讓可增強的範圍變小，不會讓模型算錯。

| 類別 | 運算 | 處理方式 |
|---|---|---|
| dense 層 | `nn.Conv2d`（`groups=1`）、`nn.Linear`（輸入必須是 2D） | 輸出開新群組 |
| member 層 | `nn.BatchNorm1d/2d/3d`、`nn.SyncBatchNorm`、depthwise `nn.Conv2d` | 輸出沿用輸入的群組 |
| 逐元素 | ReLU、ReLU6、SiLU、GELU、Hardswish、Hardsigmoid、Sigmoid、Tanh、LeakyReLU、ELU、Mish、Dropout、Identity、clamp 等（模組、`F.*`、`torch.*`、方法呼叫都算） | 穿透 |
| 空間運算 | MaxPool、AvgPool、AdaptiveAvg/MaxPool、Upsample、`F.interpolate`、torchvision `stochastic_depth` | 穿透 |
| 相加、相乘 | `+`、`-`、`*`、`/`、`torch.add` 等 | 合併兩邊的群組。只有 1 個通道的張量（例如空間注意力 `x * sigmoid(conv(x))`，其中 conv 的 `out_channels=1`）沿通道廣播，不參與合併，但它的群組會被**固定**在 1 個通道，不會被加寬 |
| 串接 | `torch.cat` | `dim=1` 串接段列表；`dim>=2` 合併群組 |
| 改形狀 | `flatten`、`view`、`reshape`、`squeeze`、`nn.Flatten` | 只在第 0、1 維都不變時穿透（例如池化後的 flatten） |
| 縮減 | `mean`、`sum`、`amax` | 只在不縮減第 0、1 維時穿透 |
| 讀取中繼資料 | `x.size()`、`x.dim()`、`x.shape` | 忽略 |

會被固定的常見情況：

- 模型輸入與輸出。
- `GroupNorm`、`LayerNorm`、`InstanceNorm`、`PReLU` 等沒有列在白名單上的模組。
- 直接參與運算的 `nn.Parameter`，例如 layer scale `x * self.gamma`。
- `1 < groups < C` 的分組卷積（例如 RegNet、ResNeXt）。
- 同一個模組被呼叫多次。
- channel shuffle、`transpose`、`chunk`、張量切片 `x[:, :k]`。
- 殘差相加的兩邊段結構不同（例如一邊是 concat、一邊是單一 conv）。
- `Conv1d`、`Conv3d`、轉置卷積（目前尚未支援）。

用 `graph.summary()` 可以看到每個群組是否被固定，以及被固定的原因：

```python
class WithGroupNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.gn = nn.GroupNorm(2, 8)          # 不在白名單上
        self.c2 = nn.Conv2d(8, 6, 1)
        self.fc = nn.Linear(6, 4)

    def forward(self, x):
        x = torch.relu(self.gn(self.c1(x)))
        return self.fc(torch.relu(self.c2(x)).mean([2, 3]))


g = analyze(WithGroupNorm(), torch.randn(1, 3, 16, 16))
assert g.group_fixed[g.layers["c1"].out_group]       # 被 GroupNorm 讀取 → 固定
assert not g.group_fixed[g.layers["c2"].out_group]   # 仍可增強
print(g.summary())
```

追蹤失敗（例如 `forward` 裡有依資料內容決定的 `if`）時，會丟出 `TraceError`。這時可以傳入自訂的 `torch.fx.Tracer`，或手動建立 `ChannelGraph`（見 README）。

## 3. 動態層：以最大寬度存權重，執行時切片

`NetAugModel` 建立時，會複製原始模型，再把 `ChannelGraph.layers` 列出的每一層換成動態層：

- 權重以**最大寬度**配置。
- 原始權重複製到 base 的位置（輸出前段、每段輸入的前段）。
- 其餘位置使用 PyTorch 預設的初始化。BN 的 weight 為 1、bias 為 0。

所以在 base 模式下，`NetAugModel` 的輸出**完全等於**原始模型（測試 `test_base_mode_equals_original` 在 float64 下以 `1e-9` 的容差驗證）。

每個動態層有兩個狀態：

| 屬性 | 意義 |
|---|---|
| `active_out` | dense 層的輸出通道數，永遠是前段切片 |
| `in_segments` | `((offset, active), ...)`：每段從 `offset` 起取 `active` 個通道。只有一段且 `offset=0` 時直接用 `narrow`（不複製）；多段時用 `index_select` |

`DynamicBatchNorm` 另有 `update_stats`：

- base 模式：正常更新 running 統計值。
- 其他模式（預設 `update_bn_in_aug=False`）：用本批統計值正規化，但**不更新** running 統計值。這樣增強網路不會污染 base 模型的統計值。

```python
import torchvision
from torchnetaug import NetAugModel, DynamicConv2d

tiny = torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10).eval()
x = torch.randn(2, 3, 32, 32)
aug = NetAugModel(tiny, x, width_mults=(1.0, 2.0, 3.0)).eval()

conv = aug.model.features[0][0]
assert isinstance(conv, DynamicConv2d)
print(conv.weight.shape, conv.active_out, conv.in_segments)  # 以最大寬度存放；base 只用前段

with torch.no_grad():
    assert torch.allclose(aug(x), tiny(x), atol=1e-5)  # base 模式等於原始模型
```

## 4. 寬度控制

`NetAugModel.set_sizes(sizes)` 為每個群組設定作用中的寬度，然後更新每個動態層的 `active_out` 與 `in_segments`。三個常用模式都建在它上面：

| 方法 | 每個群組的寬度 |
|---|---|
| `set_base()` | 原始大小 |
| `set_max()` | 最大選項 |
| `set_random(generator, granularity)` | `"group"`：每個未固定群組各自從選項中抽；`"global"`：整個模型抽同一個倍數 |

以群組為單位抽樣，自然對應到論文官方實作的粒度：ResNet 的一個 stage 是一個群組；MobileNetV2 inverted block 的擴張通道也是獨立群組，所以 expand ratio 會被一起增強。

```python
import random

rng = random.Random(0)
aug.set_random(rng)
for g, size in enumerate(aug.active_sizes):
    assert size in aug.choices[g]
aug.set_base()
```

## 5. 通道排序

### 5.1 為什麼需要排序

base 模型永遠使用每個群組的**前段**通道。訓練一段時間後，重要的通道不一定在前面。排序把每個群組中最重要的通道搬到前面，讓 base 模型拿到最有用的通道。

### 5.2 參考實作在殘差網路上失敗的原因

`netaug/`（2023 年的 YOLOv8 實作）的排序是**逐模組**進行的：每個模組依自己的權重算出排列，只重排自己的輸出通道和 BN（`dim=0`），沒有重排下游層的輸入維度（`dim=1`）。這有兩個後果：

1. **下游層沒跟著重排**：下游層的輸入權重仍對應舊的通道順序，所以網路的運算結果被改變了。
2. **殘差兩邊各排各的**：`x + f(x)` 兩邊來自不同模組，各自用不同的排列。相加時第 i 個通道其實是兩個不同的特徵相加。

所以每次排序都等於破壞一次已訓練的網路，訓練因此無法收斂。詳見 [reference/netaug_reference_impl.md](reference/netaug_reference_impl.md) 第 3.4 節。

### 5.3 本套件的做法：一個群組一個排列

排序以**通道群組**為單位，而不是以模組為單位。因為殘差相加已經把兩邊併成同一個群組，所以兩邊一定用同一個排列。

對每個未固定的群組：

1. **算重要度**（`channel_importance()`）：把所有讀取這個群組的 dense 層，在對應輸入段上的權重絕對值加總（對輸入維度以外的維度求和）。如果這個群組沒有任何 dense 讀取者，改用產生它的層的輸出權重。計算精度至少為 float32；權重是 float64 時維持 float64。
2. **排序**：`argsort(重要度, descending=True, stable=True)` 得到排列 `perm`。新通道 `i` = 舊通道 `perm[i]`。
3. **套用到所有攜帶這個群組通道的張量**（`_perm_ops()` 列出清單）：

| 張量 | 維度 | 位置 |
|---|---|---|
| 產生者（dense 層）的 `weight`、`bias` | 0 | 前段 |
| member 層（BN 的 `weight`、`bias`、`running_mean`、`running_var`；depthwise conv 的 `weight`、`bias`） | 0 | 該段的偏移量 |
| 讀取者（dense 層）的 `weight` | 1 | 該段的偏移量（concat 時不是 0） |
| 上面這些參數在 optimizer 裡、形狀相同的狀態（SGD momentum、Adam 的 `exp_avg`、`exp_avg_sq` 等） | 同參數 | 同參數 |
| `extra_models`（例如 EMA 模型）的同名張量 | 同上 | 同上 |

因為每條通道路徑的兩端都套用同一個排列，**最大寬度的網路函數完全不變**；改變的只是「較小寬度取到哪些通道」。

```python
aug.eval()
aug.set_max()
with torch.no_grad():
    before = aug(x)
perms = aug.sort_channels()            # {群組編號: 排列}
with torch.no_grad():
    after = aug(x)
assert torch.allclose(before, after, atol=1e-4)  # 最大寬度的輸出不變

for g, imp in aug.channel_importance().items():   # 排序後重要度遞減
    assert torch.all(imp[:-1] >= imp[1:] * (1 - 1e-5))
aug.set_base()
```

注意：排序**會**改變 base 模型的輸出，這是排序的目的。所以排序應該在訓練過程中定期執行，之後還要繼續訓練；不要在匯出前一刻才排序。

### 5.4 排序時的訓練狀態

- **optimizer**：傳入 `sort_channels(optimizer=opt)`。形狀和參數相同的狀態張量會套用同一個排列。測試 `test_permutation_commutes_with_optimizer_step` 驗證「先排列再走一步」等於「先走一步再排列」（SGD 與 Adam）。
- **EMA 或其他相同結構的模型**：傳入 `sort_channels(extra_models=[ema])`。它們會套用同一組排列，而不是各自重算。
- **學習率排程器**：不受影響。
- **訓練模式下的 Dropout**：遮罩作用在通道上，排序前後同一個亂數遮罩會蓋到不同的通道。這不影響正確性，只是逐位比對輸出時要關掉 Dropout。

### 5.5 驗證排序正確性的測試

| 測試 | 驗證的事 |
|---|---|
| `test_sort_keeps_max_function` | 8 種模型（含 ResNet、MobileNetV2/V3、EfficientNet、concat 模型），eval 與 train 模式下，排序前後最大寬度的輸出相同 |
| `test_sort_only_outputs_breaks_function` | 反向對照：只重排產生者的輸出（參考實作的做法），輸出**一定會**改變 |
| `test_sort_orders_importance` | 排序後每個群組的重要度遞減 |
| `test_permutation_commutes_with_optimizer_step` | optimizer 狀態跟著正確重排 |
| `test_sort_with_extra_model_keeps_ema_aligned` | EMA 模型與主模型保持一致 |
| `test_widths.py` | **每個寬度都是它宣稱的那個網路**：每種動態層在任意寬度（含 concat 的多段輸入）下，輸出等於同大小的普通 PyTorch 層；TinyResNet 在 x1.0、x1.5、x2.0、x2.5、x3.0 下，參數量、每層權重形狀、eval/train 輸出、BN 統計值更新，都等於直接用該寬度建出的獨立模型（參考模型的形狀來自模型自己的建構子，不經過通道圖） |
| `test_random_architectures.py` | **任意模組組合**：60 個隨機產生的有向無環圖，混合 dense conv、depthwise conv、BN、各種激活、池化、殘差相加、SE 相乘、concat、Linear，以及不支援的 GroupNorm、分組卷積、layer scale、通道切片、channel shuffle。每個都檢查：base 等於原始模型；依重要度排序、再套用完全隨機的排列後，最大寬度輸出都不變；匯出正確。另有一個測試確認產生器真的產出非平凡的結構（可增強群組、member 層、concat 讀取者的數量都有下限） |

這組隨機測試本身也被驗證過：把排序故意改成「只重排輸出、不重排讀取者的輸入」（參考實作的錯誤），第一個隨機架構就會失敗。

所有測試在跑之前都會把參數與 BN 統計值隨機化。預設的 BN 值對每個通道都一樣，漏排 BN 也測不出來；隨機化後就測得出來。

### 5.6 數學證明

這一節證明三件事：(1) 每個寬度都是一個普通的小網路；(2) base 寬度等於原始模型；(3) 排序不改變最大寬度的網路函數。證明只用到一個關於運算的前提（引理 S），而這個前提由測試逐一檢查。

**記號。** `torch.fx` 把模型變成一個有向無環圖。每個張量 `t` 有一個段列表 `spec(t) = (g1, …, gk)`：第 1 維（通道維）依序由群組 `g1 … gk` 的通道組成，第 i 段在最大寬度佈局中的偏移量是 `off_i`。群組 `g` 的最大寬度是 `n_g`，作用寬度是 `k_g ≤ n_g`。

**引理 S（逐通道獨立）。** 如果運算 `f` 對每個通道套用同一個函數，也就是輸出通道 `c` 只取決於輸入通道 `c`，那麼對任何索引序列 `I`（一個排列，或一個子集）都有 `f(x[:, I]) = f(x)[:, I]`。

白名單裡的每個逐元素、池化、上採樣、改形狀與縮減運算都必須滿足 S。`tests/test_whitelist_lemmas.py` 從 `graph.py` 的白名單自動產生測試案例，對每個運算檢查「隨機排列」與「隨機子集」兩種情況。白名單新增運算卻沒有對應案例時，`test_every_whitelisted_op_has_a_case` 會失敗。同一個檔案也檢查四種動態層的權重引理：`layer(x[:, P_in]; W[P_out][:, P_in]) = layer(x; W)[:, P_out]`。

**定理 1（每個寬度都是一個普通網路）。** 固定每個群組的作用寬度 `k_g`。令 `S_g` 為群組 `g` 的前 `k_g` 個通道，`S(spec) = ∪_i (off_i + S_gi)`。把每個張量都只保留 `S(spec)` 中的通道、每個權重都只保留對應的行和列，得到的就是一個普通的小網路 `N_k`。動態模型在寬度 `k` 下的輸出等於 `N_k` 的輸出。

*證明*（依拓撲順序歸納）。歸納假設：動態模型裡的每個張量，正好等於 `N_k` 中對應的張量，而且通道依 `spec` 的段順序排列。
- dense 層：輸出前 `k_g` 個通道（`active_out`）；讀取的權重列是 `off_i + [0, k_gi)`，和輸入張量的段順序一一對應。
- member 層與白名單運算：由引理 S，它們把縮減後的張量對應到縮減後的張量。
- 相加、相乘：分析時已用 union-find 把兩邊併成同一個群組，所以兩邊的 `spec` 和寬度相同。
- `torch.cat`（dim=1）：縮減後張量的串接，正好是各段依序排列。
- 其他運算：分析時已把它碰到的群組固定，所以寬度不變。
∎

`export(sizes)` 就是把 `N_k` 實際建出來：每個動態層換成作用大小的普通層。`test_widths.py` 把它和「用模型自己的建構子直接建出的同寬度模型」逐一比對。

**系理（base 等於原始模型）。** 建立 `NetAugModel` 時，原始權重被複製到每個群組的前 `base` 個通道（concat 時在各段的偏移量上）。取 `k = base`，定理 1 中的 `N_base` 就是原始網路本身。

**定理 2（排序不變最大寬度的函數）。** 對每個未固定群組 `g` 取一個排列 `P_g`；固定群組取恆等排列。對一個 `spec`，令 `Π(spec)` 為把各段的 `P_gi` 放在各自偏移量上的分塊排列。依 5.3 節的表格排列所有張量之後，最大寬度網路中的每個張量 `t` 都變成 `t[:, Π(spec(t))]`。因為模型輸出屬於固定群組，所以輸出不變。

*證明*（依拓撲順序歸納）。
- 模型輸入：固定群組，排列是恆等。
- dense 層：新權重 `W' = W[P_out][:, Π_in]`，所以 `y'[o] = Σ_i W[P_out(o), Π_in(i)] · x[Π_in(i)] = y[P_out(o)]`。等號右邊只是把同一組項重新排列後相加。
- BN 與 depthwise conv：每個通道的參數用同一個排列移動，運算逐通道獨立（引理 S）。訓練模式下的批次統計值也是逐通道計算。
- 白名單運算：引理 S。
- 相加、相乘：兩邊屬於同一群組，用同一個 `Π`。只有 1 個通道的廣播運算元屬於固定群組，不會被排列；純量沒有通道。
- `torch.cat`（dim=1）：`Π` 是各輸入 `Π` 的分塊組合。dim≥2 的串接已被併成同一群組。
- 其他運算：分析時已把它碰到的群組固定，輸入沒有被排列，所以輸出也沒有。
∎

**定理 3（optimizer 狀態）。** SGD（含 momentum、weight decay）和 Adam 的更新都是逐元素運算。對參數和狀態張量套用同一個排列，再走一步，結果等於先走一步再排列。測試 `test_permutation_commutes_with_optimizer_step` 驗證這一點。

**證明沒有涵蓋的部分。**
- 分析程式本身（union-find 與白名單規則的實作）沒有形式化驗證。它的正確性靠 8 種模型、60 個隨機架構，以及 torchvision 的多種模型實際比對來支持。
- 浮點數：定理 2 在實數下是等式。在浮點數下，相加順序改變會造成誤差：float32 下相對誤差約 `1e-6`，float64 下約 `1e-12`。
- 訓練模式的 Dropout：每個元素各自抽遮罩，所以排序前後只在機率分布上相同，逐值不相等。
- `torch.fx` 只記錄範例輸入走過的路徑；依資料決定的控制流程不在分析範圍內。
- 排序**會**改變較小寬度（包含 base）的網路。這是排序的目的，不在定理 2 的範圍內。

## 6. 匯出

`export()` 切到 base 模式，取出每個動態層的作用中切片，載入**原始模型的副本**，所以回傳的就是原始類別，形狀也和原始模型完全相同。部署端不需要安裝本套件。

```python
small = aug.export().eval()
assert type(small) is type(tiny)
assert {k: v.shape for k, v in small.state_dict().items()} == {k: v.shape for k, v in tiny.state_dict().items()}
with torch.no_grad():
    assert torch.allclose(small(x), aug(x), atol=1e-5)  # aug 已在 base 模式
```

`export(sizes)` 匯出任意寬度（定理 1 的 `N_k`）。結果保留原始模型的結構與 `forward`，每個被調整寬度的層換成同類別、作用大小的普通層。它也不依賴本套件：

```python
aug.set_width(2.0)
sizes = list(aug.active_sizes)
wide = aug.export(sizes).eval()
aug.eval()
with torch.no_grad():
    assert torch.allclose(wide(x), aug(x), atol=1e-5)
assert sum(p.numel() for p in wide.parameters()) == aug.num_active_params()
assert not any(type(m).__module__.startswith("torchnetaug") for m in wide.modules())
aug.set_base()
```

## 7. 訓練一步做了什麼

`netaug_step()` 實作論文的目標 `L(base) + alpha × L(augmented)`，每一步只取樣一個增強網路：

1. `set_base()` → forward → `backward(base_loss)`。
2. `set_random()` → forward → `backward(alpha × aug_loss)`。兩次梯度累加在共用權重上。
3. 回到 `set_base()`。`optimizer.step()` 由呼叫端執行。

base 模式下，只有前段權重有梯度；增強那次 forward 才會讓額外通道得到梯度（測試 `test_netaug_step_gradients`）。

`recalibrate_bn()` 在 base 模式下把所有 BN 的 running 統計值重設，改用累積平均（`momentum=None`）跑過指定的批次，結束後還原原本的模式、momentum 和訓練旗標。它會重設整個最大寬度的 BN，所以增強通道的統計值也會被清掉；這不影響 base 模型與匯出結果。

## 8. 與論文、參考實作的差異

| 項目 | 論文官方實作 | `netaug/` 參考實作 | `torchnetaug` |
|---|---|---|---|
| 適用模型 | 手寫的 MobileNet 系列 | 手寫的 YOLOv8 模組 | 任何可被 `torch.fx` 追蹤的模型（自動分析） |
| 增強維度 | 寬度 + expand ratio | 只有寬度 | 每個通道群組的寬度（expand ratio 是其中一種群組） |
| 寬度取樣 | 每個 stage 一次 | 每層各自 | 每個群組一次，或整體一次 |
| 排序 | 同一排列套到相連層 | 只排輸出，殘差兩邊各排 | 一個群組一個排列，套到所有相連張量與 optimizer 狀態 |
| 增強時的 BN | 會更新統計值，評估前重設 | 會更新，可選重設 | 預設不更新，另提供 `recalibrate_bn` |
| 匯出 | 手寫 `export()` | 手寫 `export_module()` | 回傳原始類別的副本 |

## 9. 已知限制

- 只支援 `nn.Conv2d`、`nn.Linear`（2D 輸入）、`BatchNorm`；其他層所在的群組會被固定，不會被增強。
- 模型必須能被 `torch.fx.symbolic_trace` 追蹤，否則要提供自訂 tracer 或手動建立 `ChannelGraph`。
- 最大寬度模型的參數量大約是 base 的 `max(width_mults)²` 倍，要注意顯示卡記憶體。
- 共用（綁定）權重的層無法增強：加寬會把共用的權重拆開。`NetAugModel` 遇到這種情況會直接丟出 `ValueError`。
- 多 GPU（DDP）尚未實際測試。
- 實際訓練效果（CIFAR 對照實驗）尚待完成。
