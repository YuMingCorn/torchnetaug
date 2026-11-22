# torchnetaug

**繁體中文** | [English](README.en.md)

在純 PyTorch 中實作 **Network Augmentation（NetAug）**。

> **專案狀態：已結束（2026-11-23）。** 測試顯示實作符合論文的定義，但在本專案的 CIFAR 實驗中，NetAug 沒有顯著提高小模型的測試準確率。見 [summary.md](summary.md)。

NetAug 出自 Han Cai 等人的論文 [*Network Augmentation for Tiny Deep Learning*](https://arxiv.org/abs/2110.08890)（ICLR 2022）。小模型的問題通常是**欠擬合**，不是過擬合。NetAug 在訓練時把小模型嵌進更寬的模型裡一起訓練，讓小模型得到額外的監督訊號。訓練完只匯出小模型，**推論成本不變**。

特色：

- **任意模型、照 PyTorch 原本的方式組合**：用一般的 `torch.nn` 層自己組網路即可。`torch.fx` 會自動分析通道結構（哪些層共用通道、殘差、concat），寬度切換、通道排序與匯出都自動處理，不必手寫動態版的模型。torchvision 的 ResNet、MobileNetV2/V3、EfficientNet、DenseNet、MnasNet 都能直接使用（測試涵蓋 ResNet18、MobileNetV2/V3、ShuffleNetV2、EfficientNet-B0；MnasNet 有匯出測試）。
- **殘差網路也能安全排序通道**：排序以「通道群組」為單位，殘差兩邊一定用同一個排列，最大寬度的網路函數保持不變。
- **匯出原始類別**：`export()` 回傳的就是你原本的模型類別，部署端不需要安裝本套件。
- **寬度與深度增強**：除了論文的寬度增強，也可以在指定的 block 後面加上額外的 block（深度增強，參考 Once-for-All 的 elastic depth）。
- **不改你的訓練迴圈**：只要把一行 `loss.backward()` 換成 `netaug_step(...)`。

程式碼設計見 [docs/design.md](docs/design.md)；實驗結果見 [docs/experiments.md](docs/experiments.md)；術語對照見 [docs/glossary.md](docs/glossary.md)。

## 安裝

需要 Python 3.10 以上、PyTorch 2.4 以上。

```bash
# 在本專案根目錄
uv sync                 # 開發環境（含 pytest、torchvision）
# 或
pip install -e .
```

## 快速開始

下面是完整的訓練流程。把假資料換成你的 DataLoader 即可。

```python
import random
import torch
import torch.nn as nn
import torchvision
from torchnetaug import NetAugModel, netaug_step, recalibrate_bn

# 1. 你的小模型
tiny = torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10)

# 2. 包成 NetAug 模型：寬度倍數必須以 1.0（原始寬度）為最小值
example = torch.randn(2, 3, 32, 32)
model = NetAugModel(tiny, example, width_mults=(1.0, 2.0, 3.0))

optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
criterion = nn.CrossEntropyLoss()
rng = random.Random(0)                      # 抽樣增強網路用的亂數

fake_loader = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(3)]

# 3. 訓練：每一步 = base loss + alpha × 一個隨機增強網路的 loss
for epoch in range(2):
    model.train()
    for images, labels in fake_loader:
        base_loss, aug_loss = netaug_step(model, images, labels, criterion, alpha=1.0, generator=rng)
        optimizer.step()
        optimizer.zero_grad()

# 4. 評估或匯出前，在 base 寬度下重新估計 BN 統計值
recalibrate_bn(model, [images for images, _ in fake_loader])

# 5. 匯出：回傳 torchvision 的 MobileNetV2，形狀和原本的 tiny 完全相同
deploy = model.export().eval()
assert type(deploy) is torchvision.models.MobileNetV2

# 6. 像平常一樣存檔與載入，不需要本套件
import os, tempfile
path = os.path.join(tempfile.mkdtemp(), "tiny_netaug.pt")
torch.save(deploy.state_dict(), path)
fresh = torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10)
fresh.load_state_dict(torch.load(path))
```

## 各模組用法

### `NetAugModel`：包裝你的模型

```python
model = NetAugModel(
    tiny,                     # 原始小模型（不會被修改）
    example,                  # 範例輸入，用來追蹤形狀；要和模型在同一個裝置上
    width_mults=(1.0, 2.0, 3.0),
    update_bn_in_aug=False,   # 預設：增強網路不更新 BN running 統計值
)
print(model.graph.summary())  # 每個通道群組的大小、是否固定、為什麼固定
```

建立時會複製模型，把可增強的層換成以最大寬度存放權重的動態層，並把原始權重複製到前段。所以在 base 模式下，輸出和原始模型完全相同：

```python
tiny.eval(); model.eval()
with torch.no_grad():
    assert torch.allclose(model(example), tiny(example), atol=1e-5)
```

常用屬性：

| 屬性 | 意義 |
|---|---|
| `model.model` | 內部的動態模型（原始模型的副本，部分層換成動態層） |
| `model.graph` | 通道分析結果 `ChannelGraph` |
| `model.base_sizes` / `model.max_sizes` | 每個群組的原始寬度與最大寬度 |
| `model.choices` | 每個群組可選的寬度 |
| `model.active_sizes` / `model.mode` | 目前的寬度與模式（`"base"`、`"max"`、`"random"`、`"custom"`） |
| `model.base_depths` / `model.max_depths` / `model.active_depths` | 每個深度增強點的額外 block 數（base 為 0；沒有用 `depth=` 時是空的 dict） |

### 自行組合網路

不需要繼承特定類別，也不需要使用本套件的層。用一般的 `torch.nn` 層，照 PyTorch 原本的方式寫 `__init__` 與 `forward`，再交給 `NetAugModel`。下面的網路同時有 concat、depthwise conv 與殘差相加：

```python
class Branches(nn.Module):
    """兩條分支在通道維度 concat。"""
    def __init__(self, c_in, c):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(c_in, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.b = nn.Sequential(nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x)], dim=1)


class SepResidual(nn.Module):
    """depthwise + pointwise，加上殘差。"""
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        return x + self.bn2(self.pw(torch.relu(self.bn1(self.dw(x)))))


class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.branches = Branches(16, 12)        # 輸出 24 通道 = 12 + 12
        self.blocks = nn.Sequential(SepResidual(24), SepResidual(24))
        self.head = nn.Linear(24, num_classes)

    def forward(self, x):
        x = self.blocks(self.branches(self.stem(x)))
        return self.head(x.mean((2, 3)))


mine = MyNet()
my_model = NetAugModel(mine, example, width_mults=(1.0, 2.0))
my_model.sort_channels()          # concat 的兩段、殘差兩邊、depthwise 都自動用一致的排列
my_deploy = my_model.export()     # 仍是 MyNet，形狀和 mine 相同
assert type(my_deploy) is MyNet
```

這裡沒有任何「每種 block 要怎麼排序」的程式碼。通道之間的關係（哪些層讀同一組通道、concat 後每段來自哪裡、殘差兩邊必須一致）由分析自動求出，排序與寬度切換都依照這個結果進行。

請**不要**自己用 `DynamicConv2d` 等動態層組網路再交給 `NetAugModel`：動態層由 `NetAugModel` 自己建立。傳入已含動態層的模型會得到 `TypeError`，訊息會指出是哪一層。

### 深度增強（`depth=`）

論文與官方程式只增強寬度。本套件另外提供深度增強，做法參考同一研究團隊的 Once-for-All（elastic depth）：在指定的 block 後面加上 `k` 個額外的 block，增強網路隨機執行其中前 `0..k` 個，base 網路一個都不執行。

```python
res = torchvision.models.resnet18(num_classes=10)
deep = NetAugModel(res, example, width_mults=(1.0, 2.0), depth={"layer1": 1, "layer3": 2})
print(deep.max_depths)            # {'layer1.1': 1, 'layer3.1': 2}

deep.set_random(rng)              # 同時抽寬度與深度
deep.set_depths({"layer3.1": 2})  # 手動指定深度，寬度不變
deep.set_base()                   # 原始 ResNet-18：沒有額外 block
assert type(deep.export()) is type(res)
```

- 鍵是模組路徑，值是額外 block 的數量。路徑指向類別正好是 `nn.Sequential` 的容器（例如 ResNet 的 stage `"layer1"`）時，代表它的最後一個 block。`nn.Sequential` 的子類別（例如 torchvision 的 `Conv2dNormActivation`）視為一個 block。沒有參數的模組（例如單獨的 ReLU）不能重複。
- 該 block 必須保持輸入的形狀（通道數與空間大小都不變），而且每次 forward 只執行一次，否則建立時會得到 `ValueError`。
- 額外 block 是該 block 的副本，權重重新初始化。若它是殘差 block，副本最後一個 BN 的權重設為 0，一開始等於恆等函數，所以加深的網路一開始和 base 網路的函數相同。
- 額外 block 的輸出和原 block 的輸出屬於同一個通道群組，所以無論額外 block 有沒有執行，後面的層讀到的通道順序都一致，排序在每個深度下都正確。
- `export()` 不帶參數時一樣回傳原始模型類別。`export(sizes, depths)` 會把每個深度增強點換成 `nn.Sequential(block, *額外 block)`，仍然是普通 PyTorch 模型。

### 切換寬度

```python
model.set_base()                          # 原始小模型（評估、匯出都用這個）
model.set_max()                           # 所有群組最寬
model.set_random(rng)                     # 每個群組各自抽一個寬度
model.set_random(rng, granularity="global")  # 整個模型抽同一個倍數
model.set_width(1.5)                      # 所有可增強群組都乘上 1.5（可以不在 width_mults 裡）
n_params = model.num_active_params()      # 目前寬度（與深度）實際用到的參數量
model.set_sizes(model.base_sizes)         # 手動指定每個群組的寬度（mode 變成 "custom"）
assert model.mode == "custom"

model.set_base()
with model.mode_ctx("max"):               # 暫時切換，離開時還原成進入前的寬度與模式
    wide_out = model(example)
assert model.mode == "base"

state = model.state()                     # (寬度, 深度, 模式)；set_state(state) 可還原
```

多 GPU 訓練時，讓每個 process 使用**相同種子**的 `random.Random`，各 GPU 才會抽到同一個增強網路。

### `netaug_step`：一個訓練步驟

```python
base_loss, aug_loss = netaug_step(
    model, images, labels, criterion,
    alpha=1.0,          # 增強 loss 的權重（論文用 1.0）
    augment=True,       # False：只訓練 base（例如訓練後段停止增強）
    generator=rng,
    granularity="group",
)
optimizer.step()
optimizer.zero_grad()
```

它只做 forward 與 backward，`optimizer.step()` 由你呼叫，所以梯度累積、梯度裁剪、學習率排程都照你原本的寫法。結束時模型會回到 base 模式。

混合精度或自訂 forward：

```python
# doc: skip  （需要 CUDA）
scaler = torch.amp.GradScaler()

def fwd(m, x):
    with torch.autocast("cuda", dtype=torch.float16):
        return m(x)

netaug_step(model, images, labels, criterion, generator=rng,
            forward=fwd, backward=lambda loss: scaler.scale(loss).backward())
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

`netaug_step` 會自動解開 `DistributedDataParallel` 的 `.module`。（DDP 尚未實際測試。）

### `sort_channels`：依重要度排序通道

定期把重要的通道搬到前面，讓 base 模型拿到最有用的通道。在 epoch 之間呼叫，並傳入 optimizer，讓 momentum 等狀態跟著一起重排：

```python
perms = model.sort_channels(optimizer=optimizer)   # 回傳 {群組編號: 排列}
```

有 EMA 模型（或其他結構相同的 `NetAugModel`）時一起傳入，它會套用同一組排列：

```python
import copy
ema = copy.deepcopy(model)
model.sort_channels(optimizer=optimizer, extra_models=[ema])
```

排序保證**最大寬度**的輸出不變，但會改變 base 的輸出（這正是排序的目的）。所以排序後要繼續訓練，不要在匯出前一刻才排序。原理見 [docs/design.md 第 5 節](docs/design.md#5-通道排序)。

### `recalibrate_bn`：重新估計 BN 統計值

```python
recalibrate_bn(model, [images for images, _ in fake_loader], num_batches=100)
```

- 預設切到 base 模式，用累積平均重新估計所有 BN 的 running 統計值，結束後還原原本的寬度與深度。
- 每個批次可以是 `(inputs, targets)` 或單一張量；其他格式請傳 `forward=lambda m, batch: m(batch["image"])`。
- 建議使用關掉資料增強的訓練資料。

評估其他寬度（例如檢查準確度是否隨寬度平緩變化）時，用 `sizes=` 指定寬度。所有寬度共用同一組 BN 緩衝區，所以要在**副本**上校正，才不會蓋掉 base 的統計值：

```python
import copy

calib = [images for images, _ in fake_loader]
for mult in (1.0, 2.0, 3.0):
    sub = copy.deepcopy(model)
    sub.set_width(mult)
    recalibrate_bn(sub, calib, sizes=sub.active_sizes)
    sub.eval()
    with torch.no_grad():
        logits = sub(example)          # 在這個寬度下評估
```

有深度增強時，用 `depths=` 指定深度。增強網路的前向傳遞不更新 BN 統計值，所以額外 block 的 BN 只有在這裡才會得到正確的統計值：

```python
deep_eval = copy.deepcopy(deep)
recalibrate_bn(deep_eval, calib, depths=deep_eval.max_depths)   # base 寬度、所有額外 block
```

### `export`：匯出小模型

```python
deploy = model.export().eval()
assert type(deploy) is type(tiny)
assert {k: v.shape for k, v in deploy.state_dict().items()} == {k: v.shape for k, v in tiny.state_dict().items()}
```

回傳原始類別的副本，裝置和 `model` 相同。之後可以像一般模型一樣 `torch.save`、`torch.onnx.export` 或 `torch.compile`。

也可以匯出其他寬度。先在那個寬度下重新校正 BN，再傳入 `sizes`：

```python
wide = copy.deepcopy(model)
wide.set_width(2.0)
recalibrate_bn(wide, calib, sizes=list(wide.active_sizes))
deploy_x2 = wide.export(wide.active_sizes).eval()   # 普通 PyTorch 模型，2 倍寬
print(sum(p.numel() for p in deploy_x2.parameters()), "parameters")
```

非 base 寬度的匯出結果保留原始模型的結構與 `forward`，只把被調整寬度的層換成同類別、作用大小的普通層。匯出的層沿用模型的 dtype（例如 bfloat16）與裝置。

有深度增強時，用 `depths=` 匯出較深的網路：

```python
deploy_deep = deep_eval.export(deep_eval.base_sizes, deep_eval.max_depths).eval()
assert not any(type(m).__module__.startswith("torchnetaug") for m in deploy_deep.modules())
```

匯出的模型可以直接轉成 ONNX（`torch.onnx.export`，dynamo 與 TorchScript 兩種匯出器都測過），onnxruntime 的輸出與 PyTorch 一致，見 `tests/test_onnx.py`。

### `analyze` 與 `ChannelGraph`：檢查或手動指定通道結構

`analyze()` 不建立模型，只回傳分析結果，可以先確認哪些部分會被增強：

```python
from torchnetaug import analyze

graph = analyze(tiny, example)
print(len(graph.free_groups()), "/", graph.num_groups, "groups can be augmented")
```

模型無法被 `torch.fx` 追蹤時（`TraceError`），可以手動建立 `ChannelGraph`，再傳給 `NetAugModel(graph=...)`：

```python
from torchnetaug import ChannelGraph, LayerRole

net = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
    nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
)
manual = ChannelGraph(
    group_sizes=[3, 8, 16, 10],              # 群組 0 輸入、1、2、3 輸出
    group_fixed=[True, False, False, True],  # 輸入與輸出固定
    layers={
        "0": LayerRole("dense", in_groups=(0,), out_group=1),
        "1": LayerRole("member", in_groups=(1,)),   # BN
        "3": LayerRole("dense", in_groups=(1,), out_group=2),
        "4": LayerRole("member", in_groups=(2,)),
        "8": LayerRole("dense", in_groups=(2,), out_group=3),
    },
)
auto = analyze(net, torch.randn(1, 3, 16, 16))
assert (auto.group_sizes, auto.group_fixed, auto.layers) == (manual.group_sizes, manual.group_fixed, manual.layers)

wrapped = NetAugModel(net, graph=manual, width_mults=(1.0, 2.0))
print(wrapped.max_sizes)   # [3, 16, 32, 10]
```

`LayerRole` 的 `kind`：

- `"dense"`：`nn.Conv2d`（`groups=1`）或 `nn.Linear`。輸出通道是 `out_group`，輸入通道依序屬於 `in_groups`（concat 時有多個）。
- `"member"`：BatchNorm 或 depthwise conv。輸入等於輸出，屬於 `in_groups`。

手動指定時，殘差相加的兩邊要使用同一個群組編號，否則排序與寬度切換都會出錯。

## 支援範圍

會被增強：`nn.Conv2d`（一般與 depthwise）、`nn.Linear`、BatchNorm，以及它們之間的常見運算（激活函數、池化、殘差相加、SE 相乘、concat、flatten）。

不會被增強（對應的通道維持原寬度，但模型仍然正確）：分組卷積、GroupNorm/LayerNorm、channel shuffle、layer scale 參數、`Conv1d`/`Conv3d` 等。完整清單見 [docs/design.md 第 2.2 節](docs/design.md#22-白名單分析)。

最大寬度模型的參數量大約是 base 的 `max(width_mults)²` 倍，例如 `width_mults=(1.0, 2.0, 3.0)` 約 9 倍。深度增強再加上額外 block 的參數。

深度增強只能用在「保持輸入形狀、每次 forward 只執行一次」的 block 上，而且需要 `torch.fx` 追蹤（不能和手動的 `graph=` 一起用）。

## 測試

```bash
uv run pytest
```

471 個測試以不經過本套件的普通 PyTorch 模型為基準，檢查 base 模式、每個寬度、一步的梯度（論文 Eq. 2）、排序與匯出。說明見 [docs/design.md 第 9 節](docs/design.md#9-正確性驗證)。

## 實驗結論

在 CIFAR-10 與 CIFAR-100（TinyResNet，40 epoch）上，NetAug 都沒有顯著提高 base 模型的測試準確率；同樣的計算量用來多訓練 40 epoch 則有 +1.54 個百分點的顯著增益。結果、觀察與重現指令見 [docs/experiments.md](docs/experiments.md)，專案總結與未來工作見 [summary.md](summary.md)。

## 引用與參考資料

本套件實作的方法出自下列論文。如果你在研究中使用 torchnetaug，請引用原論文：

```bibtex
@inproceedings{cai2022network,
    title={Network Augmentation for Tiny Deep Learning},
    author={Han Cai and Chuang Gan and Ji Lin and Song Han},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=TYw3-OlrRm-}
}
```

- 論文：<https://arxiv.org/abs/2110.08890>。本地副本在 [docs/reference/papers/](docs/reference/papers/)，授權見該資料夾的說明。
- 官方實作：<https://github.com/mit-han-lab/tinyml/tree/master/netaug>（MIT 授權）。本套件參考了它的設計（權重切片、只取樣一個增強網路、BN 重設、匯出），但程式碼是獨立撰寫的，沒有複製官方程式碼。
- 論文與官方實作的整理（已對照 PDF 原文）：[docs/reference/netaug_paper.md](docs/reference/netaug_paper.md)
- 本專案作者先前在 YOLOv8 上的實作分析：[docs/reference/netaug_reference_impl.md](docs/reference/netaug_reference_impl.md)。原始碼不在本 repo 中。
- 實驗資料集 CIFAR-10：Alex Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. Technical report, University of Toronto, 2009.

## 授權

程式碼採用 MIT 授權，見 [LICENSE](LICENSE)。

例外：[docs/reference/papers/](docs/reference/papers/) 中的論文 PDF 屬於原作者，採用 CC BY-NC-SA 4.0 授權，不適用 MIT 授權，也不會被打包進套件。
