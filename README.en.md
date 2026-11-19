# torchnetaug

[繁體中文](README.md) | **English**

A pure-PyTorch implementation of **Network Augmentation (NetAug)**, which improves the accuracy of tiny models.

NetAug comes from the paper [*Network Augmentation for Tiny Deep Learning*](https://arxiv.org/abs/2110.08890) by Han Cai et al. (ICLR 2022). The usual problem of a tiny model is **underfitting**, not overfitting. During training, NetAug embeds the tiny model in wider models and trains them together, so the tiny model gets extra supervision. After training, only the tiny model is exported, so **the inference cost does not change**.

Features:

- **Any model, composed the normal PyTorch way**: build your network from ordinary `torch.nn` layers. `torch.fx` analyzes the channel structure automatically (which layers share channels, residuals, concat). Width switching, channel sorting and export are all automatic. You do not need to write a dynamic version of the model by hand. torchvision ResNet, MobileNetV2/V3, EfficientNet, DenseNet and MnasNet work directly (tests cover ResNet18, MobileNetV2/V3, ShuffleNetV2 and EfficientNet-B0; MnasNet has an export test).
- **Safe channel sorting on residual networks too**: sorting works per "channel group". The two sides of a residual always use the same permutation, so the function of the max-width network does not change.
- **Export to the original class**: `export()` returns your original model class. The deployment side does not need this package.
- **Width and depth augmentation**: besides the width augmentation of the paper, you can add extra blocks after chosen blocks (depth augmentation, based on the elastic depth of Once-for-All).
- **No change to your training loop**: replace one line, `loss.backward()`, with `netaug_step(...)`.

For the internal design and the sorting method, see [docs/architecture.en.md](docs/architecture.en.md). For the experiment and verification report, see [docs/report.en.md](docs/report.en.md). For the Chinese–English term list, see [docs/glossary.md](docs/glossary.md).

## Installation

Requires Python 3.10 or later and PyTorch 2.4 or later.

```bash
# in the project root
uv sync                 # development environment (includes pytest, torchvision)
# or
pip install -e .
```

## Quick start

Below is the full training flow. Replace the fake data with your DataLoader.

```python
import random
import torch
import torch.nn as nn
import torchvision
from torchnetaug import NetAugModel, netaug_step, recalibrate_bn

# 1. Your tiny model
tiny = torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10)

# 2. Wrap it as a NetAug model: the smallest width multiplier must be 1.0 (the original width)
example = torch.randn(2, 3, 32, 32)
model = NetAugModel(tiny, example, width_mults=(1.0, 2.0, 3.0))

optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
criterion = nn.CrossEntropyLoss()
rng = random.Random(0)                      # random numbers for sampling augmented networks

fake_loader = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(3)]

# 3. Train: each step = base loss + alpha × the loss of one random augmented network
for epoch in range(2):
    model.train()
    for images, labels in fake_loader:
        base_loss, aug_loss = netaug_step(model, images, labels, criterion, alpha=1.0, generator=rng)
        optimizer.step()
        optimizer.zero_grad()

# 4. Before evaluation or export, re-estimate the BN statistics at base width
recalibrate_bn(model, [images for images, _ in fake_loader])

# 5. Export: returns a torchvision MobileNetV2 with exactly the same shapes as the original tiny
deploy = model.export().eval()
assert type(deploy) is torchvision.models.MobileNetV2

# 6. Save and load as usual; this package is not needed
import os, tempfile
path = os.path.join(tempfile.mkdtemp(), "tiny_netaug.pt")
torch.save(deploy.state_dict(), path)
fresh = torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10)
fresh.load_state_dict(torch.load(path))
```

## Module usage

### `NetAugModel`: wrap your model

```python
model = NetAugModel(
    tiny,                     # original tiny model (not modified)
    example,                  # example input for shape tracing; must be on the same device as the model
    width_mults=(1.0, 2.0, 3.0),
    update_bn_in_aug=False,   # default: augmented networks do not update BN running statistics
)
print(model.graph.summary())  # size of each channel group, whether it is fixed, and why
```

On construction, the wrapper copies the model, replaces the augmentable layers with dynamic layers that store weights at max width, and copies the original weights into the leading part. So in base mode, the output is exactly the same as the original model:

```python
tiny.eval(); model.eval()
with torch.no_grad():
    assert torch.allclose(model(example), tiny(example), atol=1e-5)
```

Common attributes:

| Attribute | Meaning |
|---|---|
| `model.model` | The internal dynamic model (a copy of the original model, with some layers replaced by dynamic layers) |
| `model.graph` | The channel analysis result, a `ChannelGraph` |
| `model.base_sizes` / `model.max_sizes` | The original width and max width of each group |
| `model.choices` | The width choices of each group |
| `model.active_sizes` / `model.mode` | The current width and mode (`"base"`, `"max"`, `"random"`, `"custom"`) |
| `model.base_depths` / `model.max_depths` / `model.active_depths` | The number of extra blocks at each depth point (0 for base; an empty dict when `depth=` is not used) |

### Composing your own network

You do not need to inherit from a specific class or use this package's layers. Use ordinary `torch.nn` layers, write `__init__` and `forward` the normal PyTorch way, and pass the model to `NetAugModel`. The network below has concat, a depthwise conv and a residual addition:

```python
class Branches(nn.Module):
    """Two branches concatenated along the channel dimension."""
    def __init__(self, c_in, c):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(c_in, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.b = nn.Sequential(nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x)], dim=1)


class SepResidual(nn.Module):
    """depthwise + pointwise, plus a residual."""
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
        self.branches = Branches(16, 12)        # 24 output channels = 12 + 12
        self.blocks = nn.Sequential(SepResidual(24), SepResidual(24))
        self.head = nn.Linear(24, num_classes)

    def forward(self, x):
        x = self.blocks(self.branches(self.stem(x)))
        return self.head(x.mean((2, 3)))


mine = MyNet()
my_model = NetAugModel(mine, example, width_mults=(1.0, 2.0))
my_model.sort_channels()          # the two concat segments, both residual sides and the depthwise conv all get consistent permutations
my_deploy = my_model.export()     # still a MyNet, with the same shapes as mine
assert type(my_deploy) is MyNet
```

There is no code here that says "how to sort each kind of block". The analysis finds the relations between channels automatically: which layers read the same channels, where each segment after a concat comes from, and which residual sides must match. Sorting and width switching both follow this result.

Do **not** build a network yourself from dynamic layers such as `DynamicConv2d` and then pass it to `NetAugModel`. `NetAugModel` creates the dynamic layers itself. A model that already contains dynamic layers gives a `TypeError`, and the message names the layer.

### Depth augmentation (`depth=`)

The paper and the official implementation augment only width. This package also provides depth augmentation, based on Once-for-All (elastic depth) from the same research group: it adds `k` extra blocks after a chosen block. An augmented network runs a random number `0..k` of them, starting from the first. The base network runs none of them.

```python
res = torchvision.models.resnet18(num_classes=10)
deep = NetAugModel(res, example, width_mults=(1.0, 2.0), depth={"layer1": 1, "layer3": 2})
print(deep.max_depths)            # {'layer1.1': 1, 'layer3.1': 2}

deep.set_random(rng)              # sample width and depth together
deep.set_depths({"layer3.1": 2})  # set depth by hand; width does not change
deep.set_base()                   # the original ResNet-18: no extra blocks
assert type(deep.export()) is type(res)
```

- The key is a module path, and the value is the number of extra blocks. When the path points to a container whose class is exactly `nn.Sequential` (for example, the ResNet stage `"layer1"`), it means the last block of that container. A subclass of `nn.Sequential` (for example, torchvision's `Conv2dNormActivation`) counts as one block. A module without parameters (for example, a standalone ReLU) cannot be repeated.
- The block must keep the shape of its input (the same number of channels and the same spatial size), and it must run only once per forward. Otherwise, construction gives a `ValueError`.
- An extra block is a copy of the block with re-initialized weights. If it is a residual block, the weight of the last BN in the copy is set to 0, so the copy starts as the identity function. So the deeper network starts with the same function as the base network.
- The output of an extra block belongs to the same channel group as the output of the original block. So the layers after it read the channels in the same order whether or not the extra blocks run, and sorting is correct at every depth.
- `export()` without arguments still returns the original model class. `export(sizes, depths)` replaces each depth point with `nn.Sequential(block, *extra blocks)`, which is still a plain PyTorch model.

### Switching width

```python
model.set_base()                          # the original tiny model (use this for evaluation and export)
model.set_max()                           # all groups at max width
model.set_random(rng)                     # each group samples its own width
model.set_random(rng, granularity="global")  # the whole model samples one multiplier
model.set_width(1.5)                      # multiply all augmentable groups by 1.5 (need not be in width_mults)
n_params = model.num_active_params()      # number of parameters actually used at the current width (and depth)
model.set_sizes(model.base_sizes)         # set the width of each group by hand (mode becomes "custom")
assert model.mode == "custom"

model.set_base()
with model.mode_ctx("max"):               # temporary switch; on exit, restores the width and mode from before entry
    wide_out = model(example)
assert model.mode == "base"

state = model.state()                     # (width, depth, mode); set_state(state) restores it
```

For multi-GPU training, give each process a `random.Random` with the **same seed**. Then all GPUs sample the same augmented network.

### `netaug_step`: one training step

```python
base_loss, aug_loss = netaug_step(
    model, images, labels, criterion,
    alpha=1.0,          # weight of the augmentation loss (the paper uses 1.0)
    augment=True,       # False: train only the base (e.g. stop augmentation late in training)
    generator=rng,
    granularity="group",
)
optimizer.step()
optimizer.zero_grad()
```

It only does the forward and backward passes. You call `optimizer.step()`, so gradient accumulation, gradient clipping and learning-rate schedules stay as you wrote them. At the end, the model returns to base mode.

Mixed precision or a custom forward:

```python
# doc: skip  (requires CUDA)
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

`netaug_step` unwraps the `.module` of `DistributedDataParallel` automatically. (DDP is not tested yet.)

### `sort_channels`: sort channels by importance

Move the important channels to the front at regular intervals, so the base model gets the most useful channels. Call it between epochs, and pass the optimizer so that its state (such as momentum) is reordered too:

```python
perms = model.sort_channels(optimizer=optimizer)   # returns {group index: permutation}
```

If you have an EMA model (or another `NetAugModel` with the same structure), pass it too. It gets the same permutations:

```python
import copy
ema = copy.deepcopy(model)
model.sort_channels(optimizer=optimizer, extra_models=[ema])
```

Sorting guarantees that the output at **max width** does not change, but it changes the base output (this is the purpose of sorting). So continue training after sorting. Do not sort just before export. For the theory, see [docs/architecture.en.md Section 5](docs/architecture.en.md#5-channel-sorting).

### `recalibrate_bn`: re-estimate BN statistics

```python
recalibrate_bn(model, [images for images, _ in fake_loader], num_batches=100)
```

- By default, it switches to base mode, re-estimates the running statistics of all BN layers with a cumulative average, and then restores the original width and depth.
- Each batch can be `(inputs, targets)` or a single tensor. For other formats, pass `forward=lambda m, batch: m(batch["image"])`.
- Use training data with data augmentation turned off.

To evaluate other widths (for example, to check that accuracy changes smoothly with width), set the width with `sizes=`. All widths share one set of BN buffers, so recalibrate on a **copy**. This keeps the base statistics from being overwritten:

```python
import copy

calib = [images for images, _ in fake_loader]
for mult in (1.0, 2.0, 3.0):
    sub = copy.deepcopy(model)
    sub.set_width(mult)
    recalibrate_bn(sub, calib, sizes=sub.active_sizes)
    sub.eval()
    with torch.no_grad():
        logits = sub(example)          # evaluate at this width
```

With depth augmentation, set the depth with `depths=`. The forward passes of augmented networks do not update BN statistics, so the BN layers of extra blocks get correct statistics only here:

```python
deep_eval = copy.deepcopy(deep)
recalibrate_bn(deep_eval, calib, depths=deep_eval.max_depths)   # base width, all extra blocks
```

### `export`: export the tiny model

```python
deploy = model.export().eval()
assert type(deploy) is type(tiny)
assert {k: v.shape for k, v in deploy.state_dict().items()} == {k: v.shape for k, v in tiny.state_dict().items()}
```

It returns a copy of the original class on the same device as `model`. After that, you can use `torch.save`, `torch.onnx.export` or `torch.compile` as with any model.

You can also export other widths. First recalibrate BN at that width, then pass `sizes`:

```python
wide = copy.deepcopy(model)
wide.set_width(2.0)
recalibrate_bn(wide, calib, sizes=list(wide.active_sizes))
deploy_x2 = wide.export(wide.active_sizes).eval()   # plain PyTorch model, 2× wide
print(sum(p.numel() for p in deploy_x2.parameters()), "parameters")
```

An export at a non-base width keeps the structure and `forward` of the original model. It only replaces the resized layers with plain layers of the same class at the active size. The exported layers keep the dtype (for example, bfloat16) and device of the model.

With depth augmentation, export a deeper network with `depths=`:

```python
deploy_deep = deep_eval.export(deep_eval.base_sizes, deep_eval.max_depths).eval()
assert not any(type(m).__module__.startswith("torchnetaug") for m in deploy_deep.modules())
```

You can convert the exported model to ONNX directly (`torch.onnx.export`; both the dynamo and the TorchScript exporters are tested). The onnxruntime output matches PyTorch. See `tests/test_onnx.py`.

### `analyze` and `ChannelGraph`: inspect or set the channel structure by hand

`analyze()` does not build a model. It only returns the analysis result, so you can first check which parts will be augmented:

```python
from torchnetaug import analyze

graph = analyze(tiny, example)
print(len(graph.free_groups()), "/", graph.num_groups, "groups can be augmented")
```

When `torch.fx` cannot trace the model (`TraceError`), you can build a `ChannelGraph` by hand and pass it to `NetAugModel(graph=...)`:

```python
from torchnetaug import ChannelGraph, LayerRole

net = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
    nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
)
manual = ChannelGraph(
    group_sizes=[3, 8, 16, 10],              # group 0 input, 1, 2, 3 output
    group_fixed=[True, False, False, True],  # input and output are fixed
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

The `kind` of `LayerRole`:

- `"dense"`: `nn.Conv2d` (`groups=1`) or `nn.Linear`. The output channels are `out_group`. The input channels belong to `in_groups` in order (more than one with concat).
- `"member"`: BatchNorm or a depthwise conv. The input equals the output, and both belong to `in_groups`.

When you set the structure by hand, the two sides of a residual addition must use the same group index. Otherwise, sorting and width switching both go wrong.

## Supported scope

Augmented: `nn.Conv2d` (normal and depthwise), `nn.Linear`, BatchNorm, and the common operations between them (activation functions, pooling, residual addition, SE multiplication, concat, flatten).

Not augmented (the related channels keep their original width, but the model is still correct): grouped convolution, GroupNorm/LayerNorm, channel shuffle, layer scale parameters, `Conv1d`/`Conv3d`, and others. For the full list, see [docs/architecture.en.md Section 2.2](docs/architecture.en.md#22-whitelist-analysis).

The max-width model has about `max(width_mults)²` times the parameters of the base. For example, `width_mults=(1.0, 2.0, 3.0)` gives about 9 times. Depth augmentation adds the parameters of the extra blocks.

Depth augmentation works only on blocks that "keep the input shape and run only once per forward". It also needs `torch.fx` tracing (it cannot be used together with a manual `graph=`).

## Verification

### Correctness: automated tests

```bash
uv run pytest
```

The tests compare outputs in float64 with tight tolerances. They first randomize all parameters and BN statistics, so default values cannot hide errors. They cover 8 models (a custom conv chain, a concat model, a GroupNorm model, ResNet18, MobileNetV2, MobileNetV3, ShuffleNetV2, EfficientNet-B0):

| What is verified | Test |
|---|---|
| Base-mode output equals the original model | `test_base_mode_equals_original` |
| The exported model is the original class, has the same shapes, and its output equals the base | `test_export_equals_base` |
| Max-width output does not change after sorting (covers residual, concat and SE) | `test_sort_keeps_max_function` |
| Order after sorting: each group's importance is non-increasing up to rounding, and sorting again only swaps channels tied up to rounding (fixed models, 60 random architectures, depth-augmented models) | `test_sort_orders_importance`, `assert_sorted` |
| **Any combination of modules**: on 60 randomly generated architectures, max-width output does not change after sorting or any permutation, and base and export are also correct | `test_random_architectures.py` |
| Sorting only the outputs (the method of the reference implementation) always breaks the output | `test_sort_only_outputs_breaks_function` |
| Optimizer state is reordered correctly | `test_permutation_commutes_with_optimizer_step` |
| Augmented networks do not pollute BN statistics | `test_aug_pass_does_not_touch_bn_stats` |
| Gradients flow only where they should | `test_netaug_step_gradients` |
| At each width, the parameter count, weight shapes and output equal those of a plain model built directly at that width | `tests/test_widths.py` |
| Premises of the mathematical proof: every whitelist operation acts on each channel independently (for both permutations and subsets), and the weight permutation lemma of the dynamic layers holds (proof in [architecture.en.md 5.6](docs/architecture.en.md)) | `tests/test_whitelist_lemmas.py` |
| A 1-channel spatial attention map is not widened; shared weights are rejected; MnasNet can be exported | `test_one_channel_broadcast_group_is_fixed`, etc. |
| Depth augmentation: base and export equal the original model, residual copies start as the identity function, max-width output at each depth does not change after sorting, `export(sizes, depths)` equals the dynamic network, state restore, error messages | `tests/test_depth.py` |
| After conversion to ONNX, the onnxruntime output of the exported model matches PyTorch (base, ×2, random widths) | `tests/test_onnx.py` |
| A dynamic-layer network with widths set by hand equals a plain network; a model that contains dynamic layers is rejected explicitly; `export(sizes)` keeps the dtype | `tests/test_dynamic_usage.py` |
| All examples in the documents run (Chinese and English versions) | `tests/test_docs.py` |

### Dynamic width check (fake data, no dataset download)

```bash
uv run python examples/check_widths_fake_data.py
```

This script uses the same model as the CIFAR experiment (TinyResNet). On synthetic data (10 classes; each class is one fixed random pattern plus strong noise), it checks:

1. Before and after training, at each width from x1.0 to x3.0, the parameter count and output equal those of "a plain TinyResNet built directly at that width, loaded with the same prefix weights".
2. It trains 400 steps with `netaug_step` and sorts every 100 steps. At each sort, the max-width output does not change, and the loss decreases.
3. The test accuracy at each width after its own BN recalibration, and that `export()` keeps the base accuracy and size (176,402 parameters).

Each check prints PASS or FAIL. If any check fails, the exit code is 1. Results on an RTX 3060 Ti on 2026-11-19 (all PASS):

| Width | Parameters | Max output difference from the plain model | Test accuracy |
|---|---|---|---|
| x1.0 (base) | 176,402 | 0 | 67.7% |
| x1.5 | 395,350 | 0 | 70.7% |
| x2.0 | 701,466 | 0 | 72.7% |
| x2.5 | 1,094,750 | 0 | 74.0% |
| x3.0 | 1,575,202 | 0 | 73.3% |

This only shows that the "size and computation" at each width are correct, and that every width can learn. The synthetic data is small, so the accuracy numbers do not show the effect of NetAug. For the effect, see the CIFAR-10 experiment below.

### Effect: CIFAR-10 comparison experiment

```bash
bash examples/run_cifar10_compare.sh            # baseline, NetAug, NetAug + sorting; 3 seeds each
bash examples/run_cifar10_depth.sh              # depth augmentation, width + depth augmentation; 3 seeds each
uv run python examples/summarize.py runs/cifar10_compare.jsonl runs/cifar10_depth.jsonl
```

- Each run adds one line to the jsonl file. The line records the test and train accuracy of the base model.
- NetAug runs also record the test accuracy at widths ×1.0, ×1.5, ×2.0, ×2.5 and ×3.0. Each width gets its own BN recalibration on a copy. Training never samples ×1.5 or ×2.5, so these widths check whether accuracy changes smoothly with width.
- The gap from a single run can be just noise, so compare the mean and standard deviation over 3 seeds.

Results: see "Experiment results" below.

### Applicability: CIFAR-100 comparison

```bash
bash examples/run_cifar100_regime.sh            # 4 settings, 3 seeds each
uv run python examples/summarize.py runs/cifar100_regime.jsonl
```

- The same TinyResNet underfits more on CIFAR-100 (100 classes). The paper predicts: the more a model underfits, the larger the NetAug gain.
- 4 settings: baseline, baseline with learning rate 0.2, NetAug with width multipliers 1/2/3, NetAug with width multipliers 1/1.2/1.4. The baseline with learning rate 0.2 checks whether NetAug only doubles the base gradient (see [architecture.en.md Section 8.1](docs/architecture.en.md)).
- `--grad-log-every N`: every N steps, record the cosine similarity and the norm ratio of the base gradient and the augmented gradient on the base weights.
- `--amp {bf16,fp16,off}`: the precision of the training forward pass. Use `fp16` on a GPU without bfloat16 (for example the T4). Results with different precisions are listed separately in the tables.

## Experiment results

CIFAR-10, TinyResNet (176,402 parameters), 40 epochs, 3 seeds (0, 1, 2) per configuration. See [docs/report.en.md](docs/report.en.md) for the full setup and discussion.

**Base width (the deployed model)**

| Method | Test acc (%) | Train acc (%) | Paired difference vs. baseline (points) |
|---|---|---|---|
| baseline | 86.92 ± 0.35 | 91.90 ± 0.23 | — |
| netaug | 87.07 ± 0.39 | 92.70 ± 0.20 | +0.15 ± 0.74 |
| netaug+sort10 | 87.26 ± 0.09 | 92.81 ± 0.23 | +0.34 ± 0.44 |

**Each width: exported from the NetAug model vs. trained alone (test acc %)**

| Width | Parameters | Trained alone | netaug exported | netaug+sort10 exported |
|---|---|---|---|---|
| ×1.0 | 176,402 | 86.92 ± 0.35 | 87.07 ± 0.39 | 87.26 ± 0.09 |
| ×1.5 | 395,350 | 89.27 ± 0.11 | 87.48 ± 0.25 | 88.00 ± 0.13 |
| ×2.0 | 701,466 | 90.46 ± 0.27 | 88.22 ± 0.16 | 88.48 ± 0.21 |
| ×3.0 | 1,575,202 | 91.62 ± 0.30 | 88.44 ± 0.18 | 88.50 ± 0.27 |

- The gain for the base model is small, smaller than the variation between seeds; 3 seeds are not enough to show an effect. Train accuracy rises by 0.7–1.3 points for every seed, which matches the "reduce underfitting" mechanism.
- Wider exported models are worse than models of the same width trained alone; NetAug targets the base model.
- Sorting never moved a channel into the base (a result of the initialization scale, see section 4 of the report), so `+sort10` has no effect on the base model.
- Every exported model has an output difference of 0 from the dynamic network.

**Depth augmentation** (1 extra block after the last block of each stage, `--depth 1`)

| Method | Test acc (%) | Train acc (%) | Paired difference vs. baseline (points) |
|---|---|---|---|
| Depth augmentation | 86.87 ± 0.20 | 92.28 ± 0.16 | −0.05 ± 0.29 |
| Width + depth augmentation | 86.96 ± 0.05 | 92.77 ± 0.21 | +0.04 ± 0.34 |

- Depth augmentation has no measurable effect on the base model, and adding depth does not make width augmentation better.
- The exported network with every extra block on (274,802 parameters) reaches 86.92 ± 0.17, the same as base: the extra blocks are residual copies that start as the identity function and train only when sampled, so they learn almost nothing in 40 epochs.

## Citation and references

The method in this package comes from the paper below. If you use torchnetaug in your research, please cite the original paper:

```bibtex
@inproceedings{cai2022network,
    title={Network Augmentation for Tiny Deep Learning},
    author={Han Cai and Chuang Gan and Ji Lin and Song Han},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=TYw3-OlrRm-}
}
```

- Paper: <https://arxiv.org/abs/2110.08890>. A local copy is in [docs/reference/papers/](docs/reference/papers/). For its license, see the notes in that folder.
- Official implementation: <https://github.com/mit-han-lab/tinyml/tree/master/netaug> (MIT license). This package follows its design (weight slicing, sampling only one augmented network, BN reset, export), but the code is written independently. No official code is copied.
- Summary of the paper and the official implementation (checked against the original PDF): [docs/reference/netaug_paper.md](docs/reference/netaug_paper.md)
- Analysis of the project author's earlier implementation on YOLOv8: [docs/reference/netaug_reference_impl.md](docs/reference/netaug_reference_impl.md). The source code is not in this repo.
- Experiment dataset CIFAR-10: Alex Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. Technical report, University of Toronto, 2009.

## License

The code is under the MIT license. See [LICENSE](LICENSE).

Exception: the paper PDFs in [docs/reference/papers/](docs/reference/papers/) belong to their original authors and are under the CC BY-NC-SA 4.0 license. The MIT license does not apply to them, and they are not packaged.
