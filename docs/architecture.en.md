# torchnetaug Architecture and Channel Sorting

[繁體中文](architecture.md) | **English**

This document describes the internal design of `torchnetaug`: how it turns any PyTorch model into a model with switchable widths, and why channel sorting is also correct on residual networks. For usage, see the [README](../README.en.md).

`tests/test_docs.py` runs every Python example in this document, so the document and the code stay in sync.

## 1. Overview

NetAug trains the tiny model (base) embedded in wider models at the same time. All widths share one set of weights. The tiny model uses the leading channels. After training, only the tiny model is exported.

The package has five files. The data flow is:

```text
original model ──► graph.analyze() ──► ChannelGraph ──► NetAugModel ──► export() ──► tiny model of the original class
                   (torch.fx tracing)   (channel groups)  (dynamic model at max width)
                                                            │
                                         train.netaug_step ─┤  each step: base loss + alpha × aug loss
                                         train.recalibrate_bn  re-estimate BN statistics before evaluation or export
                                         NetAugModel.sort_channels  sort channels by importance
```

| File | Contents |
|---|---|
| `src/torchnetaug/graph.py` | `analyze()`, `ChannelGraph`, `LayerRole`, `TraceError`: find the channel groups |
| `src/torchnetaug/layers.py` | `DynamicConv2d`, `DynamicLinear`, `DynamicBatchNorm`: store weights at max width and slice them at run time |
| `src/torchnetaug/depth.py` | `DepthStack`, `tie_channels`: extra blocks for depth augmentation (Section 7) |
| `src/torchnetaug/netaug.py` | `NetAugModel`: build the max-width model, switch width and depth, sort, export |
| `src/torchnetaug/train.py` | `netaug_step()`, `recalibrate_bn()`: training helpers |

## 2. Core data structure: the channel group

A **channel group** is a set of channels that must have the same width and the same channel order. Examples:

- The output channels of a dense conv, and the input channels of every layer that reads it.
- The two sides of a residual addition `x + f(x)`. So the outputs of all blocks in one ResNet stage merge into **one** group.
- The input and output of a batch norm or a depthwise conv.

Each group has three properties:

- **Size**: the number of channels in the original model. This is the base width.
- **Fixed or not**: a fixed group always keeps its original width and is never sorted. The model input, the model output, and every place that cannot be analyzed safely are fixed.
- **Width choices**: `round(size × m)`, where `m` comes from `width_mults`. The largest choice is the max width.

Each dynamic layer has one `LayerRole` in `ChannelGraph.layers`:

| `kind` | Layer type | Meaning |
|---|---|---|
| `"dense"` | `nn.Conv2d` (`groups=1`), `nn.Linear` | The output channels form their own group `out_group`; the input channels belong to `in_groups` in order |
| `"member"` | `nn.BatchNorm*`, depthwise `nn.Conv2d` | Input channels equal output channels, and both belong to `in_groups` |

### 2.1 concat and segments

`torch.cat(..., dim=1)` does not merge groups. Its output is a **list of segments**. Each concatenated group takes one segment. For example, a DenseNet-style concatenation:

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
        y = self.bn(torch.cat([x0, x1], dim=1))  # 10 = 6 + 4 channels, two segments
        return self.head(y.mean([2, 3]))


graph = analyze(DenseLike(), torch.randn(1, 3, 16, 16))
roles = graph.layers
stem_g, c1_g = roles["stem"].out_group, roles["c1"].out_group
assert roles["bn"].in_groups == (stem_g, c1_g)    # BN reads two segments
assert roles["head"].in_groups == (stem_g, c1_g)  # Linear reads the same two segments
print(graph.summary())
```

In the max-width weights, each segment has its own fixed position (offset). The offset is the sum of the **max** widths of all earlier segments. At run time, each segment uses only its first `active` channels:

```text
input dimension of head.weight (max width = 18 + 12 = 30)
┌───────── stem segment: offset 0, max 18 ───────────┬──── c1 segment: offset 18, max 12 ────┐
│ ■■■■■■ (base 6) □□□□□□□□□□□□                       │ ■■■■ (base 4) □□□□□□□□                │
└────────────────────────────────────────────────────┴──────────────────────────────────────┘
indices used in base mode = [0..5] ∪ [18..21]
```

### 2.2 Whitelist analysis

`analyze()` traces the model with `torch.fx.symbolic_trace`. Then it uses `ShapeProp` to compute the tensor shape at each node. Finally it applies the rules node by node. **Only operations on the whitelist let channels pass through.** Every other operation fixes the groups it touches. This is a conservative approach: an unknown operation only makes the augmentable part smaller. It never makes the model compute wrong results.

| Category | Operations | Handling |
|---|---|---|
| dense layers | `nn.Conv2d` (`groups=1`), `nn.Linear` (input must be 2D) | The output starts a new group |
| member layers | `nn.BatchNorm1d/2d/3d`, `nn.SyncBatchNorm`, depthwise `nn.Conv2d` | The output keeps the input's group |
| element-wise | ReLU, ReLU6, SiLU, GELU, Hardswish, Hardsigmoid, Sigmoid, Tanh, LeakyReLU, ELU, Mish, Dropout, Identity, clamp, and others (modules, `F.*`, `torch.*`, and method calls all count) | Pass through |
| spatial | MaxPool, AvgPool, AdaptiveAvg/MaxPool, Upsample, `F.interpolate`, torchvision `stochastic_depth` | Pass through |
| add, multiply | `+`, `-`, `*`, `/`, `torch.add`, and others | Merge the groups of both sides. A tensor with only 1 channel (for example spatial attention `x * sigmoid(conv(x))`, where the conv has `out_channels=1`) broadcasts along the channels and does not take part in the merge. But its group is **fixed** at 1 channel and is not widened |
| concatenation | `torch.cat` | `dim=1` concatenates segment lists; `dim>=2` merges groups |
| reshape | `flatten`, `view`, `reshape`, `squeeze`, `nn.Flatten` | Pass through only when dimensions 0 and 1 stay unchanged (for example flatten after pooling) |
| reduction | `mean`, `sum`, `amax` | Pass through only when dimensions 0 and 1 are not reduced |
| metadata reads | `x.size()`, `x.dim()`, `x.shape` | Ignored |

Common cases that become fixed:

- The model input and output.
- Modules that are not on the whitelist, such as `GroupNorm`, `LayerNorm`, `InstanceNorm`, and `PReLU`.
- An `nn.Parameter` used directly in a computation, for example layer scale `x * self.gamma`.
- Grouped convolutions with `1 < groups < C` (for example RegNet and ResNeXt).
- A module that is called more than once.
- channel shuffle, `transpose`, `chunk`, and tensor slicing `x[:, :k]`.
- A residual addition whose two sides have different segment structures (for example a concat on one side and a single conv on the other).
- `Conv1d`, `Conv3d`, and transposed convolutions (not supported yet).

`graph.summary()` shows whether each group is fixed, and why:

```python
class WithGroupNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.gn = nn.GroupNorm(2, 8)          # not on the whitelist
        self.c2 = nn.Conv2d(8, 6, 1)
        self.fc = nn.Linear(6, 4)

    def forward(self, x):
        x = torch.relu(self.gn(self.c1(x)))
        return self.fc(torch.relu(self.c2(x)).mean([2, 3]))


g = analyze(WithGroupNorm(), torch.randn(1, 3, 16, 16))
assert g.group_fixed[g.layers["c1"].out_group]       # read by GroupNorm → fixed
assert not g.group_fixed[g.layers["c2"].out_group]   # still augmentable
print(g.summary())
```

When tracing fails (for example, because `forward` contains an `if` that depends on the data), `analyze()` raises `TraceError`. In that case, you can pass a custom `torch.fx.Tracer`, or build a `ChannelGraph` by hand (see the README).

## 3. Dynamic layers: store weights at max width, slice at run time

When you create a `NetAugModel`, it copies the original model. Then it replaces every layer listed in `ChannelGraph.layers` with a dynamic layer:

- The weights are allocated at **max width**.
- The original weights are copied to the base positions (the leading output channels, and the leading channels of each input segment).
- All other positions use the PyTorch default initialization. BN weight is 1 and bias is 0.

So in base mode, the output of `NetAugModel` is **exactly equal** to the original model (the test `test_base_mode_equals_original` checks this in float64 with a tolerance of `1e-9`).

Each dynamic layer has two state attributes:

| Attribute | Meaning |
|---|---|
| `active_out` | The number of output channels of a dense layer. It is always a leading slice |
| `in_segments` | `((offset, active), ...)`: each segment takes `active` channels starting at `offset`. With one segment and `offset=0`, the layer uses `narrow` directly (no copy); with several segments, it uses `index_select` |

`DynamicBatchNorm` also has `update_stats`:

- Base mode: the running statistics update normally.
- Other modes (default `update_bn_in_aug=False`): the layer normalizes with the batch statistics but does **not** update the running statistics. This way the augmented network does not corrupt the statistics of the base model.

```python
import torchvision
from torchnetaug import NetAugModel, DynamicConv2d

tiny = torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10).eval()
x = torch.randn(2, 3, 32, 32)
aug = NetAugModel(tiny, x, width_mults=(1.0, 2.0, 3.0)).eval()

conv = aug.model.features[0][0]
assert isinstance(conv, DynamicConv2d)
print(conv.weight.shape, conv.active_out, conv.in_segments)  # stored at max width; base uses only the leading part

with torch.no_grad():
    assert torch.allclose(aug(x), tiny(x), atol=1e-5)  # base mode equals the original model
```

## 4. Width control

`NetAugModel.set_sizes(sizes)` sets the active width of each group. Then it updates `active_out` and `in_segments` of every dynamic layer. The three common modes are built on it:

| Method | Width of each group |
|---|---|
| `set_base()` | Original size |
| `set_max()` | Largest choice |
| `set_random(generator, granularity)` | `"group"`: each free group samples its own choice; `"global"`: the whole model samples one multiplier |

Sampling per group matches the granularity of the paper's official implementation. One ResNet stage is one group. The expansion channels of a MobileNetV2 inverted block are also a separate group, so the expand ratio is augmented as well.

```python
import random

rng = random.Random(0)
aug.set_random(rng)
for g, size in enumerate(aug.active_sizes):
    assert size in aug.choices[g]
aug.set_base()
```

## 5. Channel sorting

### 5.1 Why sorting is needed

The base model always uses the **leading** channels of each group. After some training, the important channels are not always at the front. Sorting moves the most important channels of each group to the front, so the base model gets the most useful channels.

### 5.2 Why the reference implementation fails on residual networks

`netaug/` (the 2023 YOLOv8 implementation) sorts **per module**. Each module computes a permutation from its own weights. It reorders only its own output channels and BN (`dim=0`). It does not reorder the input dimension (`dim=1`) of downstream layers. This has two consequences:

1. **Downstream layers are not reordered**: the input weights of downstream layers still match the old channel order, so the network computes a different result.
2. **The two residual sides are sorted separately**: the two sides of `x + f(x)` come from different modules, and each uses a different permutation. In the addition, channel i is actually the sum of two different features.

So each sort breaks the trained network, and training cannot converge. For details, see Section 3.4 of [reference/netaug_reference_impl.md](reference/netaug_reference_impl.md).

### 5.3 This package's approach: one permutation per group

Sorting works per **channel group**, not per module. The residual addition already merges both sides into one group, so both sides always use the same permutation.

For each free group:

1. **Compute importance** (`channel_importance()`): sum the absolute values of the weights of every dense layer that reads this group, over the matching input segment (sum over all dimensions except the input dimension). If the group has no dense reader, use the output weights of the layer that produces it. The computation uses at least float32; float64 weights stay in float64.
2. **Sort**: `argsort(importance, descending=True, stable=True)` gives the permutation `perm`. New channel `i` = old channel `perm[i]`.
3. **Apply it to every tensor that carries this group's channels** (`_perm_ops()` lists them):

| Tensor | Dimension | Position |
|---|---|---|
| `weight` and `bias` of the producer (dense layer) | 0 | Leading part |
| member layers (BN `weight`, `bias`, `running_mean`, `running_var`; depthwise conv `weight`, `bias`) | 0 | Offset of the segment |
| `weight` of a reader (dense layer) | 1 | Offset of the segment (not 0 for concat) |
| Optimizer state with the same shape as these parameters (SGD momentum, Adam `exp_avg`, `exp_avg_sq`, and others) | Same as the parameter | Same as the parameter |
| Tensors with the same name in `extra_models` (for example an EMA model) | Same as above | Same as above |

Both ends of every channel path get the same permutation, so **the function of the max-width network does not change at all**. The only change is which channels the smaller widths use.

```python
aug.eval()
aug.set_max()
with torch.no_grad():
    before = aug(x)
perms = aug.sort_channels()            # {group index: permutation}
with torch.no_grad():
    after = aug(x)
assert torch.allclose(before, after, atol=1e-4)  # max-width output does not change

for g, imp in aug.channel_importance().items():   # importance decreases after sorting
    assert torch.all(imp[:-1] >= imp[1:] * (1 - 1e-5))
aug.set_base()
```

Note: sorting **does** change the output of the base model. This is the purpose of sorting. So run sorting periodically during training, and continue training after it. Do not sort just before export.

### 5.4 Training state during sorting

- **Optimizer**: pass `sort_channels(optimizer=opt)`. State tensors with the same shape as their parameter get the same permutation. The test `test_permutation_commutes_with_optimizer_step` checks that "permute, then step" equals "step, then permute" (SGD and Adam).
- **EMA or other models with the same structure**: pass `sort_channels(extra_models=[ema])`. They get the same permutations; they do not compute their own.
- **Learning rate schedulers**: not affected.
- **Dropout in training mode**: the mask acts on channels. Before and after sorting, the same random mask covers different channels. This does not affect correctness. But turn off Dropout when you compare outputs value by value.

### 5.5 Tests that verify sorting

| Test | What it checks |
|---|---|
| `test_sort_keeps_max_function` | For 8 models (including ResNet, MobileNetV2/V3, EfficientNet, and concat models), in eval and train mode, the max-width output is the same before and after sorting |
| `test_sort_only_outputs_breaks_function` | Negative control: reordering only the producer outputs (the reference implementation's approach) **always** changes the output |
| `test_sort_orders_importance` | Order after sorting (`assert_sorted`): the recomputed importance of each group is non-increasing up to rounding, and sorting again only swaps channels whose importance is equal up to rounding. The same check also runs on the 60 random architectures and the depth-augmented models |
| `test_permutation_commutes_with_optimizer_step` | The optimizer state is reordered correctly |
| `test_sort_with_extra_model_keeps_ema_aligned` | The EMA model stays consistent with the main model |
| `test_widths.py` | **Each width is the network it claims to be**: at any width (including multi-segment concat inputs), each dynamic layer type gives the same output as a plain PyTorch layer of the same size. At x1.0, x1.5, x2.0, x2.5, and x3.0, TinyResNet matches an independent model built directly at that width in parameter count, the weight shape of each layer, eval/train output, and BN statistic updates (the reference model's shapes come from the model's own constructor, not from the channel graph) |
| `test_random_architectures.py` | **Any combination of modules**: 60 randomly generated directed acyclic graphs that mix dense conv, depthwise conv, BN, various activations, pooling, residual addition, SE multiplication, concat, Linear, and the unsupported GroupNorm, grouped convolution, layer scale, channel slicing, and channel shuffle. For each one, the test checks that base equals the original model; that the max-width output does not change after sorting by importance and after applying a fully random permutation; and that export is correct. Another test confirms that the generator really produces non-trivial structures (the numbers of free groups, member layers, and concat readers each have a lower bound) |

The random tests themselves were also verified. When sorting was deliberately changed to "reorder only the outputs, not the readers' inputs" (the bug in the reference implementation), the first random architecture failed.

Why the order holds: after sorting, each channel's importance is the sum of the same values as before sorting, so the channels are in non-increasing order of importance. The recomputed values are not bit-identical, though: the permutations of the other groups change the order in which the readers' weights are summed, which gives rounding errors (no violation in any float64 test; at most about 1.6e-6 relative error measured in float32). So the exact statement is: non-increasing up to rounding, and sorting again only swaps channels whose importance is equal up to rounding (`argsort(stable=True)`). `assert_sorted` checks exactly this, with tolerance `rtol` (default 1e-9, as the tests run in float64; use about 1e-5 for float32). On float32, on CPU and CUDA, ResNet-18 and EfficientNet-B0 with 6 seeds each all pass. When sorting is changed to ascending order or to a random permutation, `assert_sorted` fails.

Every test randomizes the parameters and BN statistics before it runs. The default BN values are the same for every channel, so a missing BN reorder would not be detected. After randomization, it is detected.

### 5.6 Mathematical proof

This section proves three things: (1) each width is a plain small network; (2) the base width equals the original model; (3) sorting does not change the function of the max-width network. The proof uses only one assumption about the operations (Lemma S), and tests check this assumption one operation at a time.

**Notation.** `torch.fx` turns the model into a directed acyclic graph. Each tensor `t` has a segment list `spec(t) = (g1, …, gk)`: dimension 1 (the channel dimension) consists of the channels of groups `g1 … gk` in order, and the offset of segment i in the max-width layout is `off_i`. The max width of group `g` is `n_g`, and its active width is `k_g ≤ n_g`.

**Lemma S (channel-wise independence).** If an operation `f` applies the same function to each channel, that is, output channel `c` depends only on input channel `c`, then for any index sequence `I` (a permutation or a subset), `f(x[:, I]) = f(x)[:, I]`.

Every element-wise, pooling, upsampling, reshape, and reduction operation on the whitelist must satisfy S. `tests/test_whitelist_lemmas.py` generates test cases automatically from the whitelist in `graph.py`. For each operation, it checks two cases: a random permutation and a random subset. If an operation is added to the whitelist without a matching case, `test_every_whitelisted_op_has_a_case` fails. The same file also checks the weight lemma for the four dynamic layers: `layer(x[:, P_in]; W[P_out][:, P_in]) = layer(x; W)[:, P_out]`.

**Theorem 1 (each width is a plain network).** Fix the active width `k_g` of each group. Let `S_g` be the first `k_g` channels of group `g`, and `S(spec) = ∪_i (off_i + S_gi)`. Keep only the channels in `S(spec)` of each tensor, and only the matching rows and columns of each weight. The result is a plain small network `N_k`. The output of the dynamic model at width `k` equals the output of `N_k`.

*Proof* (by induction in topological order). Induction hypothesis: each tensor in the dynamic model equals the matching tensor in `N_k` exactly, and its channels are in the segment order of `spec`.
- dense layer: it outputs the first `k_g` channels (`active_out`). The weight columns it reads are `off_i + [0, k_gi)`, which match the segment order of the input tensor one to one.
- member layers and whitelisted operations: by Lemma S, they map reduced tensors to reduced tensors.
- add, multiply: the analysis already merged both sides into one group with union-find, so both sides have the same `spec` and width.
- `torch.cat` (dim=1): the concatenation of the reduced tensors is exactly the segments in order.
- other operations: the analysis already fixed the groups they touch, so the width does not change.
∎

`export(sizes)` builds `N_k` as a real model: it replaces each dynamic layer with a plain layer of the active size. `test_widths.py` compares it, one by one, with a model of the same width built directly by the model's own constructor.

**Corollary (base equals the original model).** When a `NetAugModel` is created, the original weights are copied to the first `base` channels of each group (at the offset of each segment for concat). Take `k = base`. Then `N_base` in Theorem 1 is the original network itself.

**Theorem 2 (sorting does not change the max-width function).** Choose a permutation `P_g` for each free group `g`; fixed groups get the identity permutation. For a `spec`, let `Π(spec)` be the block permutation that places each segment's `P_gi` at its offset. After all tensors are permuted as in the table in Section 5.3, each tensor `t` in the max-width network becomes `t[:, Π(spec(t))]`. The model output belongs to a fixed group, so the output does not change.

*Proof* (by induction in topological order).
- model input: a fixed group, so the permutation is the identity.
- dense layer: the new weight is `W' = W[P_out][:, Π_in]`, so `y'[o] = Σ_i W[P_out(o), Π_in(i)] · x[Π_in(i)] = y[P_out(o)]`. The right side only sums the same terms in a different order.
- BN and depthwise conv: the per-channel parameters move with the same permutation, and the operation is channel-wise independent (Lemma S). In training mode, the batch statistics are also computed per channel.
- whitelisted operations: Lemma S.
- add, multiply: both sides belong to the same group and use the same `Π`. A broadcast operand with only 1 channel belongs to a fixed group and is not permuted; a scalar has no channels.
- `torch.cat` (dim=1): `Π` is the block combination of the input `Π`s. A concatenation with dim≥2 was already merged into one group.
- other operations: the analysis already fixed the groups they touch. The input is not permuted, so the output is not permuted either.
∎

**Theorem 3 (optimizer state).** The updates of SGD (with momentum and weight decay) and Adam are element-wise operations. Applying the same permutation to the parameters and state tensors and then taking a step gives the same result as taking a step and then permuting. The test `test_permutation_commutes_with_optimizer_step` checks this.

**What the proof does not cover.**
- The analysis code itself (the implementation of union-find and the whitelist rules) is not formally verified. Its correctness is supported by direct comparisons on 8 models, 60 random architectures, and many torchvision models.
- Floating point: Theorem 2 is an equality over the real numbers. In floating point, a different summation order causes error: about `1e-6` relative error in float32, and about `1e-12` in float64.
- Dropout in training mode: each element samples its own mask, so outputs before and after sorting are the same only in distribution, not value by value.
- `torch.fx` records only the path that the example input takes. Data-dependent control flow is outside the scope of the analysis.
- Sorting **does** change the networks at smaller widths (including base). This is the purpose of sorting, and it is outside the scope of Theorem 2.

## 6. Export

`export()` switches to base mode, takes the active slice of each dynamic layer, and loads it into **a copy of the original model**. So it returns the original class, with exactly the same shapes as the original model. The deployment side does not need this package.

```python
small = aug.export().eval()
assert type(small) is type(tiny)
assert {k: v.shape for k, v in small.state_dict().items()} == {k: v.shape for k, v in tiny.state_dict().items()}
with torch.no_grad():
    assert torch.allclose(small(x), aug(x), atol=1e-5)  # aug is already in base mode
```

`export(sizes)` exports any width (`N_k` in Theorem 1). The result keeps the structure and `forward` of the original model. Each layer whose width changed is replaced with a plain layer of the same class and the active size. It also does not depend on this package:

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

## 7. Depth augmentation

The paper and the official code augment only width (the paper's reason: on GPUs, widening costs less training time than deepening). The depth augmentation in this package follows the elastic depth of Once-for-All (Cai et al., ICLR 2020), from the same research group: a stage keeps its leading blocks, and deeper subnetworks also run the later blocks. Mapped to NetAug: the base network is the original network, and the augmented network runs `0..k` extra blocks after each specified block.

### 7.1 Construction

Before tracing, `NetAugModel(..., depth={path: k})` replaces each specified block in the model copy with `DepthStack(block, [copy × k])`:

```text
DepthStack.forward(x):
    y = block(x)
    for i in range(active):              # when active = 0, this is just block(x)
        y = tie_channels(extra[i](y), y)
    return y
```

- Before building, it runs the block once on the example input. It checks that the block is called only once, and that the output shape equals the input shape. The copies read the block's output, so the two shapes must be the same.
- The weights of the copies are re-initialized (`reset_parameters()`). Then it tries to set the weight and bias of the last BN in each copy to 0. If this makes the copy an identity function on the example input (a residual block), the zeros are kept. Otherwise (a non-residual block, where zeros would cut the signal), it uses the normal re-initialization.
- All later analysis, widening, and sorting work on this copy that contains `DepthStack` (during tracing, `active` is at its maximum, so the analysis sees all extra blocks). A separate copy of the original model is kept for `export()`.

### 7.2 Why `tie_channels` is needed

Let the block's output belong to group B, and let `L` be the layer after the block. During tracing, all extra blocks run, so `L` is recorded as a reader of "the output of the last extra block". But in base mode, `L` actually reads the output of the block itself.

- Residual block: the output is `x + f(x)`. The addition merges the input and output into one group, so the output of an extra block naturally belongs to B.
- Non-residual block (for example Conv-BN-ReLU): the output of an extra block is a new group C. Without special handling, B and C get different permutations during sorting. The input weights of `L` are reordered by the permutation of C, but in base mode `L` reads channels ordered by the permutation of B. The network function then changes without any error message.

`tie_channels(x, ref)` returns `x` directly at run time. But it is registered with `torch.fx.wrap`, so it leaves a node after tracing. When the analyzer sees this node, it merges the groups of `x` and `ref` (their sizes must be equal). So the outputs of all extra blocks belong to B, and the channels that `L` reads use the same permutation at every depth.

The test `test_sorting_keeps_the_max_width_network_at_every_depth` checks that at max width, the output for every depth combination (including all zeros) is the same before and after sorting. When `tie` is changed so that it does not merge, this test fails for non-residual networks.

```python
from torchnetaug import DepthStack


class PlainBlock(nn.Sequential):
    def __init__(self, c):
        super().__init__(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())


plain = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
    PlainBlock(8), PlainBlock(8),                       # non-residual; depth augmentation on the last one
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10),
)
deep = NetAugModel(plain, x, width_mults=(1.0, 2.0), depth={"4": 2}).eval()
assert isinstance(deep.model[4], DepthStack)
producer = deep.graph.layers["4.block.0"].out_group
assert all(deep.graph.layers[f"4.extra.{i}.0"].out_group == producer for i in range(2))

outs = []
for d in (0, 1, 2):
    deep.set_state((deep.max_sizes, {"4": d}, "max"))
    with torch.no_grad():
        outs.append(deep(x))
deep.sort_channels()
for d, before in zip((0, 1, 2), outs):
    deep.set_state((deep.max_sizes, {"4": d}, "max"))
    with torch.no_grad():
        assert torch.allclose(deep(x), before, atol=1e-5)
deep.set_base()
```

### 7.3 State, BN, and export

- Width and depth form one state `(sizes, depths, mode)`. `state()` / `set_state()` read and write it. `mode_ctx`, `export(sizes, depths)`, and `recalibrate_bn` all restore the full state when they finish. `is_base` is true only when both width and depth are base. Only then does BN update its statistics.
- `set_random()` samples the widths first. Then it samples one number uniformly from `0..k` for each depth point. Without `depth=`, it uses no extra random numbers, so width-only training draws the same sequence as earlier versions.
- Extra blocks run only in the augmented network, and by default the augmented network does not update BN statistics. So before you evaluate or export a deeper network, calibrate it with `recalibrate_bn(..., depths=...)`.
- `export()` (base) renames the keys `P.block.*` back to `P.*`, drops `P.extra.*`, and loads the result into a copy of the original model. So the result is exactly the same as without depth augmentation. `export(sizes, depths)` replaces each `DepthStack` with `block` or `nn.Sequential(block, *active extra blocks)`.
- `num_active_params()` does not count extra blocks that do not run. It matches the parameter count of `export(sizes, depths)`.

### 7.4 Only plain layers are accepted

`analyze()` and `NetAugModel` both require a tiny model built from plain `torch.nn` layers. If the model already contains a `DynamicModule` (for example, the `.model` of another `NetAugModel`, or a network you built from dynamic layers), you get a `TypeError` that names the module path and class. The reasons:

- Tracing treats dynamic layers as unknown operations, so all channel groups become fixed.
- If you bypass tracing with a manual `graph=`, the dynamic layers are not widened, and `export()` keeps the dynamic layers. Deployment then requires this package.

When you build your own network, use only plain layers. `NetAugModel` creates the dynamic layers from the analysis result.

## 8. What one training step does

`netaug_step()` implements the paper's objective `L(base) + alpha × L(augmented)`. Each step samples only one augmented network:

1. `set_base()` → forward → `backward(base_loss)`.
2. `set_random()` → forward → `backward(alpha × aug_loss)`. The two gradients accumulate on the shared weights.
3. Return to `set_base()`. The caller runs `optimizer.step()`.

In base mode, only the leading weights get gradients. Only the augmented forward pass gives gradients to the extra channels (test `test_netaug_step_gradients`).

The accumulated gradient equals the paper's Eq. 2 element by element: `∂L(base)/∂W + alpha × ∂L(aug)/∂W`. The test `test_netaug_step_gradient_is_paper_eq2` checks this with two independent plain models: `export()` (base) and `export(sizes)` (the augmented network sampled in the same step). Each one computes its own gradients. The test puts them back at their positions in the shared weights and adds them. The sum must equal the result of `netaug_step` (float64, 8 models × 3 seeds).

`recalibrate_bn()` resets the running statistics of all BN layers in base mode. Then it runs the given batches with a cumulative average (`momentum=None`). At the end, it restores the original width and depth, the momentum, and the training flag of every module (also when an error occurs). It resets the full max-width BN, so it also clears the statistics of the augmented channels. This does not affect the base model or the export result. The exception is an extra block that does not run at the calibrated depth: its BN is not reset and keeps its statistics.

### 8.1 What gradient the base weights receive

The paper says only that the augmented network gives "extra supervision". It does not derive what this gradient does. Below, a two-layer linear network shows its structure. This is **a derivation of the gradient's structure, not a proof of its effect**: it shows what NetAug changes, but it does not guarantee a higher test accuracy.

Let the base network be `y = V·W·x`. The augmented network adds extra channels `W_i` (new outputs of the first layer) and `V_i` (the second-layer weights that read these new channels):

```text
base:               y  = V·W·x
augmented network:  y' = V·W·x + V_i·W_i·x
```

With squared error `L = ½‖y − t‖²`, the residuals are `r = y − t` and `r' = y' − t = r + V_i·W_i·x`. The gradients with respect to the base weights `W` are:

```text
g_base = Vᵀ · r  · xᵀ
g_aug  = Vᵀ · r' · xᵀ = g_base + Vᵀ · (V_i·W_i·x) · xᵀ

NetAug (alpha = 1): g_base + g_aug = 2·g_base + Vᵀ · (V_i·W_i·x) · xᵀ
```

The expression has two parts:

1. **`2·g_base`**: the base gradient doubled. For SGD, this is close to doubling the learning rate of the base weights. So part of "NetAug works" may only be a larger step for the base weights. The paper's ablations do not rule this out (Table 7 tests weight decay, data augmentation, and batch size, but not the learning rate).
2. **`Vᵀ·(V_i·W_i·x)·xᵀ`**: the part that is specific to NetAug. It makes `W` work together with the extra channels: what the extra channels already predict, `W` learns less of.

A non-linear network has one more effect. In the augmented network, the layers downstream of the base channels also read the extra channels. So the ReLU on/off pattern and the BN batch statistics change, and `g_aug` is no longer `g_base` plus one term. For cross-entropy, a first-order expansion around `y` gives the same structure.

The weight of the two parts can be measured. `examples/cifar_netaug.py --grad-log-every N` records, every N steps, the cosine similarity of `g_base` and `g_aug` on the base weights and the norm ratio `‖g_aug‖ / ‖g_base‖`. A cosine near 1 and a ratio near 1 mean that, in this setting, NetAug is almost the same as a larger learning rate. To rule out this explanation, a control group is needed: the baseline with twice the learning rate.

## 9. Differences from the paper and the reference implementation

| Item | Paper's official implementation | `netaug/` reference implementation | `torchnetaug` |
|---|---|---|---|
| Supported models | Hand-written MobileNet family | Hand-written YOLOv8 modules | Any model that `torch.fx` can trace (automatic analysis) |
| Augmented dimensions | Width + expand ratio | Width only (`max_depth` is unused) | The width of each channel group (expand ratio is one kind of group), and optional depth (Section 7) |
| Width sampling | The first block of each stage, the stem, and the head each sample a width (residual blocks follow); each block samples its expand ratio | Each layer separately | Once per group, or once for the whole model |
| Sorting | Only the middle channels of inverted blocks; the same permutation applied to connected layers | Outputs only; the two residual sides are sorted separately | One permutation per group, applied to all connected tensors and the optimizer state |
| BN during augmentation | Updates statistics; resets them before evaluation | Updates; optional reset | No update by default; `recalibrate_bn` is provided separately |
| Export | Hand-written `export()` | Hand-written `export_module()` | Returns a copy of the original class |

## 10. Known limitations

- Only `nn.Conv2d`, `nn.Linear` (2D input), and `BatchNorm` are supported. Groups that contain other layers are fixed and are not augmented.
- The model must be traceable by `torch.fx.symbolic_trace`. Otherwise, you must provide a custom tracer or build a `ChannelGraph` by hand.
- The max-width model has about `max(width_mults)²` times the parameters of the base model. Watch the GPU memory.
- Layers with shared (tied) weights cannot be augmented: widening would split the shared weights. `NetAugModel` raises `ValueError` in this case.
- Multi-GPU training (DDP) is not tested yet.
- Depth augmentation works only on blocks that keep the shape and run once per forward pass. It cannot be used together with a manual `graph=`.
- For the actual training results, see [report.en.md](report.en.md): the gain on CIFAR-10 is small, and 3 seeds are not enough to draw a conclusion.
