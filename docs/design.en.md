# torchnetaug code design

[繁體中文](design.md) | **English**

For usage, see the [README](../README.en.md). For experiments, see [experiments.en.md](experiments.en.md). For terms, see [glossary.md](glossary.md). `tests/test_docs.py` runs the Python examples in this document.

## Key points

1. **Channel groups are the core**: `analyze()` uses `torch.fx` to find the channels that must have the same width and order. Width switching, sorting and export all work per group (Section 2).
2. **Dynamic layers store weights at max width**: the weights of the base model are at the front, so the output in base mode equals the original model (Section 3).
3. **One permutation per group**: sorting does not change the function of the max-width network (Section 5, with proof).
4. **Export returns the original class**: any width can be exported as a plain PyTorch model (Section 6).
5. **Correctness has a test chain**: 609 tests compare each link with plain PyTorch models (Section 9).

## 1. Overview

```mermaid
flowchart LR
    M["original model"] -- "analyze()<br/>torch.fx" --> G["ChannelGraph<br/>channel groups"]
    G --> N["NetAugModel<br/>dynamic model"]
    N -- "netaug_step" --> N
    N -- "sort_channels" --> N
    N -- "recalibrate_bn" --> N
    N -- "export()" --> E["base model of<br/>the original class"]
```

| File | Content |
|---|---|
| `graph.py` | `analyze()`, `ChannelGraph`, `LayerRole`, `TraceError`: find the channel groups |
| `layers.py` | `DynamicConv2d`, `DynamicLinear`, `DynamicBatchNorm`: dynamic layers |
| `depth.py` | `DepthStack`, `tie_channels`: depth augmentation |
| `netaug.py` | `NetAugModel`: build the dynamic model, switch width and depth, sort, export |
| `train.py` | `netaug_step()`, `recalibrate_bn()` |

## 2. Channel groups

A **channel group** is a set of channels that must have the same width and the same order. These channels belong to one group:

- The output of a dense layer and the inputs of all layers that read it.
- The two sides of a residual addition `x + f(x)`. So the outputs of all blocks in one ResNet stage are one group.
- The input and output of BN and of a depthwise conv.

Each group has a size (the base width), a fixed flag, and width choices `round(size × m)` with `m` from `width_mults`. A fixed group is not widened and not sorted. Model inputs, model outputs, and places that cannot be analyzed safely are fixed.

Each dynamic layer has a `LayerRole`:

| `kind` | Layer | Meaning |
|---|---|---|
| `"dense"` | `nn.Conv2d` (`groups=1`), `nn.Linear` | The output channels are `out_group`; the input channels belong to `in_groups` in order |
| `"member"` | `nn.BatchNorm*`, depthwise `nn.Conv2d` | Input equals output and belongs to `in_groups` |

### 2.1 Concat and segments

`torch.cat(..., dim=1)` does not merge groups. It makes a **segment list**, with one segment for each concatenated group:

```mermaid
flowchart LR
    S["stem<br/>producer → group A"] --> C1["c1<br/>producer → group B"]
    S --> CAT["cat(dim=1)<br/>segments [A | B]"]
    C1 --> CAT
    CAT --> BN["bn<br/>reads [A | B]"]
    BN --> H["head<br/>reads [A | B]"]
```

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

In the max-width weights, the offset of a segment is the sum of the **max** widths of the segments before it. At run time each segment uses only its first `active` channels (■ is a base channel, □ is an added channel):

```text
input dimension of head.weight (max width 18 + 12 = 30)
┌──── stem segment: offset 0, max 18 ────┬──── c1 segment: offset 18, max 12 ────┐
│ ■■■■■■ (base 6) □□□□□□□□□□□□           │ ■■■■ (base 4) □□□□□□□□                │
└────────────────────────────────────────┴───────────────────────────────────────┘
indices in base mode = [0..5] ∪ [18..21]
```

### 2.2 Whitelist analysis

`analyze()` traces the model with `torch.fx.symbolic_trace`, gets shapes with `ShapeProp`, and applies rules node by node. **Only whitelist operations let channels pass through.** Any other operation fixes the groups it touches. So an unknown operation only makes the augmentable part smaller. It never makes the model compute a wrong result.

| Category | Operations | Handling |
|---|---|---|
| Dense layers | `nn.Conv2d` (`groups=1`), `nn.Linear` (2D input) | The output starts a new group |
| Member layers | `nn.BatchNorm1d/2d/3d`, depthwise `nn.Conv2d` | Keeps the input group (`nn.SyncBatchNorm` is fixed) |
| Element-wise | ReLU, SiLU, GELU, Hardswish, Sigmoid, Tanh, Dropout, Identity, `clamp`, and others (modules, `F.*`, `torch.*`, method calls) | Pass through; `clamp` with per-channel tensor bounds is fixed |
| Spatial | MaxPool, AvgPool, Adaptive*Pool, Upsample, `F.interpolate`, `stochastic_depth` | Pass through |
| Add, multiply | `+`, `-`, `*`, `/`, `torch.add`, and others | Merge the groups of both sides; a 1-channel tensor broadcasts over channels and its group is fixed at 1 |
| Concat | `torch.cat` | `dim=1` concatenates segment lists; `dim>=2` merges groups |
| Reshape | `flatten`, `view`, `reshape`, `squeeze`, `nn.Flatten` | Pass through when dims 0 and 1 are kept at any width; otherwise fixed |
| Reduce | `mean`, `sum`, `amax` | Pass through when dims 0 and 1 are not reduced |
| Metadata | `x.size()`, `x.dim()`, `x.shape` | Ignored |

Common reasons for a fixed group:

| Reason | Examples |
|---|---|
| Module not on the whitelist | GroupNorm, LayerNorm, PReLU, `Conv1d`, `Conv3d`, transposed convolution |
| Parameter used directly in an operation | The `nn.Parameter` of a layer scale |
| Changed channel mapping | Grouped convolution with `1 < groups < C`, channel shuffle, `transpose`, `chunk`, channel slicing |
| Structure conflict | A module called more than once; residual sides with different segment structures |

`graph.summary()` lists why each group is fixed:

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

If tracing fails (for example, an `if` that depends on the data), `analyze()` raises `TraceError`. Then pass a custom `torch.fx.Tracer` or build a `ChannelGraph` by hand.

## 3. Dynamic layers

`NetAugModel` copies the original model and replaces the layers in `ChannelGraph.layers` with dynamic layers:

- The weights are allocated at max width. The original weights are copied to the base positions.
- The other positions use the PyTorch default initialization (BN weight 1, bias 0).
- So in base mode the output **equals** the original model.

| Attribute | Meaning |
|---|---|
| `active_out` | Number of output channels of a dense layer; always the first channels |
| `in_segments` | `((offset, active), ...)`: each segment takes `active` channels from `offset`; one segment with `offset=0` uses `narrow`, otherwise `index_select` |

`DynamicBatchNorm` updates the running statistics only in base mode. In an augmented network (default `update_bn_in_aug=False`), BN normalizes with batch statistics but does not update the running statistics, so the base model is not polluted.

```python
import torchvision
from torchnetaug import NetAugModel, DynamicConv2d

tiny = torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10).eval()
x = torch.randn(2, 3, 32, 32)
aug = NetAugModel(tiny, x, width_mults=(1.0, 2.0, 3.0)).eval()

conv = aug.model.features[0][0]
assert isinstance(conv, DynamicConv2d)
print(conv.weight.shape, conv.active_out, conv.in_segments)  # stored at max width; base uses the first part

with torch.no_grad():
    assert torch.allclose(aug(x), tiny(x), atol=1e-5)  # base mode equals the original model
```

## 4. Width control

`set_sizes(sizes)` sets the width of each group and updates all dynamic layers. `set_base()`, `set_max()` and `set_random(generator, granularity)` are built on it:

- `granularity="group"`: each free group samples its own width. This matches the granularity of the official implementation: one ResNet stage is one group, and the expand channels of MobileNetV2 are separate groups.
- `granularity="global"`: the whole model samples one multiplier.

```python
import random

rng = random.Random(0)
aug.set_random(rng)
for g, size in enumerate(aug.active_sizes):
    assert size in aug.choices[g]
aug.set_base()
```

## 5. Channel sorting

The base model always uses the first channels. Sorting moves the most important channels of each group to the front.

**Why sort per group.** Suppose a layer permutes only its own outputs (`dim=0`) and not the inputs of the downstream layers (`dim=1`). Or suppose the two sides of a residual use different permutations. Then each sort changes the network function. This package gives each group exactly one permutation. A residual has already merged its two sides into one group, so both sides always use the same permutation.

How each operation handles a permutation:

| Operation | How the permutation is handled |
|---|---|
| Dense layer (conv with `groups=1`, linear) | The permutation stops here: the input uses the permutation of the input group (dim 1), and the output uses the permutation of the output group (dim 0) |
| Member layer (BN, depthwise conv) and channel-wise whitelist operations | The permutation passes through: input and output are the same group |
| Add, multiply (residual, SE scaling) | Both sides are merged into one group, so they use the same permutation |
| `cat(dim=1)` | Each segment keeps its own group and permutation |

So sorting does not need depthwise layers. It needs only this: every producer, member layer and reader of a group uses the same permutation. The ResNet bottleneck of OFA has no depthwise layer, and OFA still sorts its two middle channel groups ([`dynamic_layers.py` L766–L841](https://github.com/mit-han-lab/once-for-all/blob/f03b2673db313b9167e2a1c2b7a5cad540cc1313/ofa/imagenet_classification/elastic_nn/modules/dynamic_layers.py#L766-L841)).

For each group in the sort scope (`scope`, see Section 5.2):

1. Compute the importance (`channel_importance()`): the L1 norm of the weights of all dense readers on the matching input segment. If there is no dense reader, use the output weights of the dense producers; BN and depthwise weights are never used. The extra blocks of depth augmentation count only for a group that lives in the extra blocks (see the importance definition in Section 5.1). The precision is at least float32.
2. `perm = argsort(importance, descending=True, stable=True)`.
3. Apply `perm` to every tensor that carries this group:

| Tensor | Dim | Position |
|---|---|---|
| `weight`, `bias` of the producer | 0 | first channels |
| Member layers (BN parameters and running statistics, depthwise conv) | 0 | segment offset |
| `weight` of the readers | 1 | segment offset |
| `.grad` of the parameters above (when not None), and optimizer state with the same shape (momentum, `exp_avg`, ...) | same as parameter | same as parameter |
| Tensors with the same names in `extra_models` (such as EMA) | same | same |

Both ends of every channel path use the same permutation, so **the function of the max-width network does not change**. Only the channels that smaller widths keep change.

```python
aug.eval()
aug.set_max()
with torch.no_grad():
    before = aug(x)
perms = aug.sort_channels()            # {group index: permutation}
with torch.no_grad():
    after = aug(x)
assert torch.allclose(before, after, atol=1e-4)  # output of the max-width network does not change

for g, imp in aug.channel_importance().items():   # importance decreases after sorting
    assert torch.all(imp[:-1] >= imp[1:] * (1 - 1e-5))
aug.set_base()
```

Sorting **does** change the output of the base model, so run it periodically during training and keep training afterwards. A Dropout mask in training mode acts on channels, so turn Dropout off for exact comparisons.

### 5.1 Proof of correctness

**Notation.** Each tensor `t` has a segment list `spec(t) = (g1, …, gk)`, and segment i has offset `off_i`. Group `g` has max width `n_g` and active width `k_g ≤ n_g`.

**Definition of importance.** Reader `L` has weight `W_L` (dim 0 is the output channels, dim 1 is the input channels) and input segment list `(g_1, …, g_m)`, where segment `i` has offset `off_i`. The reader set `R_g` of group `g` is the dense layers that read `g`. If some of them are outside the extra blocks, `R_g` contains only those. So only a group whose readers are all in the extra blocks (a group that lives in the extra blocks) is ranked by the layers of the extra blocks.

`imp_g[c] = Σ_{L ∈ R_g} Σ_{i: g_i = g} Σ_{o, other dims} abs(W_L[o, off_i + c, …])`, with `c ∈ [0, n_g)`, where `o` runs over all output channels of `W_L` (max width).

If `R_g` is empty, dim 0 of the dense producers' `W_L` is used instead (summed over all input columns), and the producers are chosen with the same extra-block rule.

**Ordered.** Group `g` is **ordered** if, for every `k` from 1 to the number of channels of the group (so also every width choice in `choices[g]`), the first `k` channels are the `k` channels with the highest importance: `min(imp_g[:k]) ≥ max(imp_g[k:])`. This is equivalent to `imp_g` being non-increasing. In floating point the comparison uses a relative tolerance (1e-9 in float64), because a permutation changes the summation order.

**Lemma S (channel independence).** If output channel `c` of `f` depends only on input channel `c`, then `f(x[:, I]) = f(x)[:, I]` for any index sequence `I` (a permutation or a subset). Every element-wise, pooling, upsampling, reshape and reduce operation on the whitelist must satisfy S. `tests/test_whitelist_lemmas.py` generates cases from the whitelist and checks both permutations and subsets. It also checks the weight lemma of the dynamic layers, `layer(x[:, P_in]; W[P_out][:, P_in]) = layer(x; W)[:, P_out]`. The test fails when an operation is added to the whitelist without a case.

**Theorem 1 (every width is a plain network).** Fix `k_g`. Keep only the channels `S(spec) = ∪_i (off_i + [0, k_gi))` of each tensor and the matching weight rows and columns. The result is a plain small network `N_k`. The dynamic model at width `k` outputs the same as `N_k`. Proof by induction in topological order: a dense layer outputs its first `k_g` channels, and the weight columns it reads match the input segments one to one; member layers and whitelist operations follow from Lemma S; the two sides of an add or multiply were merged into one group by union-find; `cat(dim=1)` is exactly the segments in order; other operations have fixed groups. ∎ `export(sizes)` builds `N_k`.

**Corollary.** With `k = base`, `N_base` is the original network.

**Theorem 2 (sorting does not change the function of the max-width network).** Choose a permutation `P_g` for each free group and the identity for fixed groups. After all tensors are permuted as in the table above, each tensor `t` of the max-width network becomes `t[:, Π(spec(t))]`, where `Π` is the block permutation of the segments' `P_gi`. The model output belongs to a fixed group, so it does not change. Proof by induction in topological order: for a dense layer, `W' = W[P_out][:, Π_in]` gives `y'[o] = y[P_out(o)]`; BN and depthwise conv move their parameters per channel (batch statistics in training mode are also per channel); both sides of an add are one group; the `Π` of `cat` is the block composition; other operations have fixed groups. ∎

**Lemma I (how importance behaves under permutations).** Choose a permutation `P_h` for each free group `h` (the identity for a group that is not sorted), apply them to all tensors as in the table above, and call the new importance `imp'`. Then for every group `g`: `imp'_g[c] = imp_g[P_g(c)]`. That is, (a) the permutations of the other groups do not change `imp_g`; (b) the group's own permutation only reorders `imp_g`.

**Proof.** `R_g` depends only on the names and roles of the layers, not on the permutations, so it is the same set before and after. Fix a reader `L ∈ R_g` and a segment with `g_i = g`. After the permutation, `W'_L[o, off_i + c] = W_L[P_out(o), off_i + P_g(c)]`, where `P_out` is the permutation of the output group of `L` (the identity for a fixed group).

- In the sum over `o`, `P_out` is a bijection on all output channels, so the sum does not change. This step does not need the output group to differ from `g`: **it also holds when the output group is `g`** (for example the conv in `x + conv(x)`), because `P_g` acting on dim 0 is still a bijection.
- The permutation of another segment `j` (`g_j ≠ g`) moves elements only inside its own column range `[off_j, off_j + n_{g_j})`. It does not touch the columns of segment `i`.
- **When `cat` repeats the same group** (`g_i = g_j = g`, `i ≠ j`), both segments are permuted by the same `P_g`, so their sum is still `imp_g[P_g(c)]`.
- Member layers do not enter `imp_g`, so they do not affect the result.
- The fallback (dim 0 of the dense producers) works the same way: in the sum over the input columns, the permutation of the input group is a bijection on the columns. ∎

**Corollary (one pass gives order, and it is a fixed point).** `sort_channels` first computes the importance of all groups, then applies all permutations `P_g = argsort(imp_g, decreasing, stable)` at once. By Lemma I, after sorting `imp'_g = imp_g ∘ P_g` is non-increasing, so every sorted group is ordered. A second sort gives the identity for every group (a stable sort keeps the order of equal elements), so the model does not change. In floating point, channels with nearly equal importance can swap, but by Theorem 2 the function of the max-width network still does not change. A group outside `scope` gets the identity, and the conclusion still holds. Test: `tests/test_sorting.py` permutes the other groups on 60 random architectures; the relative change of the importance is ≤ 1e-9.

**Theorem 3 (optimizer state and `.grad`).** The updates of SGD (with momentum, nesterov and weight decay), Adam and AdamW act on each parameter element independently. If the loss of every step is computed at max width, then by Theorem 2 the gradient of the loss with respect to the parameters is equivariant under the permutation, so "sort, then take k steps" equals "take k steps, then sort", for both the parameters and the optimizer state. Steps of the base model and of augmented widths are not covered: sorting changes the output at smaller widths, so the later gradients differ anyway. What Theorem 3 shows is that permuting the optimizer state keeps each state element with its parameter element. `tests/test_sorting.py` checks this for k ∈ {1, 5, 20}; the relative deviation is < 1e-12.

- `apply_permutations` also permutes the `.grad` of the parameters (when not None), so sorting between `backward()` and `step()` is also correct.
- If the state is not permuted, the momentum and the parameter channels do not match. In a **toy experiment**, the parameters after sorting and further training deviate by 21–31% (relative); rebuilding the optimizer (dropping the momentum) deviates by 31–58%; permuting the state deviates by about 1e-15.
- The official implementation does not read or write the optimizer state when it sorts: the optimizer is built once before training ([`train.py` L198–L203](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L198-L203)), and sorting runs every epoch ([`train.py` L260–L263](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/train.py#L260-L263)). The NetAug implementation in tinyml has no code that reads or writes `optimizer.state`.
- OFA sorts at the start of each phase, before the first optimizer step ([`progressive_shrinking.py` L353–L354](https://github.com/mit-han-lab/once-for-all/blob/f03b2673db313b9167e2a1c2b7a5cad540cc1313/ofa/imagenet_classification/elastic_nn/training/progressive_shrinking.py#L353-L354)), so it does not have this problem.

**Numerical tolerance.** The outputs after sorting are compared by relative error. In train mode, BN uses the batch statistics of only 2 images, which amplifies float64 rounding errors. So the train-mode tolerance is relaxed to 1e-8 (`test_sort_one_kind`) and 1e-7 (`test_sort_keeps_max_function`); the eval-mode tolerance stays at 1e-10 and 1e-8.

**Not covered.**

- The analysis code itself (union-find and the whitelist rules) is not formally verified. Tests support it.
- In floating point, a different summation order gives a relative error of about `1e-6` (float32) or `1e-12` (float64).
- `torch.fx` records only the path that the example input takes.

### 5.2 Group kinds and sort scope

`group_kinds()` puts each group into one of 6 kinds. The checks run in the order of the diagram below, and the first match wins:

```mermaid
flowchart TD
    A["group"] --> F{"fixed?"}
    F -- "yes" --> KF["fixed"]
    F -- "no" --> D{"layers both inside and<br/>outside an extra block?"}
    D -- "yes" --> KD["depth-tied"]
    D -- "no" --> I{"all layers inside<br/>one block?"}
    I -- "yes" --> KI["inner"]
    I -- "no" --> C{"one segment of a<br/>multi-segment input?"}
    C -- "yes" --> KC["concat-segment"]
    C -- "no" --> R{"two or more producers?"}
    R -- "yes" --> KR["residual"]
    R -- "no" --> KL["closed"]
```

| Kind | Condition | Examples |
|---|---|---|
| `fixed` | A fixed group | Input image, classifier output |
| `depth-tied` | The group has layers both inside and outside an extra block | Stage trunk with `depth=` |
| `inner` | All layers of the group are inside one block | Expand channels of MBv2, SE hidden channels, middle channels of BasicBlock and Bottleneck |
| `concat-segment` | The group is one segment of the input of a reader or member layer (that input has several segments) | Branch outputs of Inception- and DenseNet-like models |
| `residual` | Two or more producers (merged into one group by an add or a multiply) | Stage trunk |
| `closed` | Any other free group: one producer | The stem and the last 1×1 conv of MBv2 |

**Automatic block detection.** With `blocks=None` (the default), `group_kinds()` finds the blocks from the module hierarchy:

| Step | Rule |
|---|---|
| Candidate | A module that is not the root, not a pure container (`nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`, `DepthStack`), and contains analyzed layers |
| Contained group | A group whose layers are all inside the candidate |
| Crossing group | A group with layers both inside and outside the candidate |
| Block | A candidate with at least one contained group, where every candidate with contained groups inside it has at most one crossing group |
| Limitation | The rule looks only at the module hierarchy. If the result is not what you want, give module paths or classes with `blocks=` |

The last rule has two effects. `InvertedResidual` contains an SE, and the SE has only one crossing group (the expand channels), so `InvertedResidual` is a block. A `backbone` that holds several stages is not a block, because a width-changing BasicBlock inside it has two crossing groups.

**Measured results** (torchvision default width, `num_classes=10`):

| Model | inner | residual | closed | fixed | Layers changed by an inner group |
|---|---|---|---|---|---|
| `mobilenet_v2` | 16 | 5 | 4 | 2 | 5 |
| `resnet18` | 8 | 4 | 0 | 2 | 3 |
| `resnet50` | 32 | 4 | 1 | 2 | 3 |
| `mobilenet_v3_small` | 19 | 5 | 3 | 2 | 2 (SE hidden), 5, 7 (expand with SE) |

**Sort scope.** `sort_channels(scope=...)` chooses the groups to sort. Groups outside the scope do not change at all (they get the identity).

| `scope` | Groups that are sorted |
|---|---|
| `"all"` (default) | Every free group |
| `"inner"` | `inner` groups; this matches the scope of the official implementation and OFA (Section 5.3) |
| `"closed"` | `inner` and `closed` groups |
| A collection of group indices | The given free groups; a fixed group raises `ValueError` |

`blocks=` is passed to `group_kinds()` for `"inner"` and `"closed"`.

```python
from collections import Counter

kinds = aug.group_kinds()                       # {group index: GroupInfo}
print(Counter(info.kind for info in kinds.values()))
first = next(info for info in kinds.values() if info.kind == "inner")
print(first.block, first.readers)               # path of the block and the dense readers

perms = aug.sort_channels(scope="inner")        # sort only the inner groups
assert set(perms) == {g for g, info in kinds.items() if info.kind == "inner"}
```

### 5.3 The official sort scope and why

The official implementation augments 6 kinds of width: the two stem outputs, the stage outputs, the expand channels, the final conv of the head, and the SE hidden channels, which change with the expand channels. It sorts only 2 of them, and both are `inner`:

| Width | Sorted by the official implementation? | Link |
|---|---|---|
| Expand channels of an inverted block | Sorted | [`utils.py` L171–L187](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L171-L187) |
| SE hidden channels | Sorted | [`utils.py` L154–L168](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L154-L168) |
| Stem, stage trunk, final conv of the head | Augmented but not sorted: `all_blocks` holds only the blocks of each stage and the inverted block of the head, and `sort_channels` is called only on these blocks | [`mbv2.py` L141–L148](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/mbv2.py#L141-L148), [`mbv2.py` L275–L277](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/mbv2.py#L275-L277); the width lists are in [`mbv2.py` L28–L139](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/mbv2.py#L28-L139) |
| Other block types | `raise NotImplementedError` | [`utils.py` L186–L187](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L186-L187) |

**Reason (inference).** The sorting code has the same structure as `re_organize_middle_weights` in OFA ([`dynamic_layers.py` L347–L413](https://github.com/mit-han-lab/once-for-all/blob/f03b2673db313b9167e2a1c2b7a5cad540cc1313/ofa/imagenet_classification/elastic_nn/modules/dynamic_layers.py#L347-L413)). In OFA-MBv3, the expand channels are the only variable width inside a block. NetAug added augmentation of the output widths, but the sorting was not extended with it. This is inferred from the code structure; the official implementation does not explain it.

This package sorts every free group by default (`scope="all"`). `scope="inner"` matches the scope of the official implementation and OFA.

**Difference in importance.** The official implementation looks at only one reader, the point conv of the block ([`utils.py` L174–L177](https://github.com/mit-han-lab/tinyml/blob/0e58f314dfe12967d6165f45238b1ea77b0f7a14/netaug/models/netaug/utils.py#L174-L177)). This package adds the raw L1 of all dense readers, so a reader with a large kernel has more weight: in a **toy network**, one 3×3 reader makes up 50% of the sum. When an `inner` group has only one dense reader, the two are equal element by element (a test checks MBv2). In a block with SE, the expand group has two dense readers (the point conv and the first conv of the SE), so the result differs from the official one.

For an item-by-item comparison with line links, see [official_diff.en.md](official_diff.en.md).

## 6. Export

`export()` switches to base mode, takes the active slices of the dynamic layers, and loads them into **a copy of the original model**. So it returns the original class with the same shapes, and the deployment side does not need this package.

`export(sizes)` exports any width (`N_k`). It keeps the original structure and `forward`, and replaces each widened layer with a plain layer of the same class and the active size. The new layer keeps the dtype, device, `.training` flag and `requires_grad` of the original layer.

```python
small = aug.export().eval()
assert type(small) is type(tiny)
assert {k: v.shape for k, v in small.state_dict().items()} == {k: v.shape for k, v in tiny.state_dict().items()}
with torch.no_grad():
    assert torch.allclose(small(x), aug(x), atol=1e-5)  # aug is already in base mode

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

The paper and the official implementation augment only width. This package follows the elastic depth of Once-for-All (Cai et al., ICLR 2020): the base model is the original network, and an augmented network runs `0..k` extra blocks after chosen blocks.

`NetAugModel(..., depth={path: k})` replaces each chosen block with a `DepthStack` before tracing:

```mermaid
flowchart LR
    X["x"] --> B["block"]
    B -- "active = 0 (base model)" --> Y["output"]
    B --> E1["extra[0]"]
    E1 -- "active = 1" --> Y
    E1 --> E2["extra[1]"]
    E2 -- "active = 2" --> Y
```

- Before building, the block is run once to check that it is called once and that its output shape equals its input shape.
- Each extra block is a copy with re-initialized weights. If setting its last BN to zero makes the copy an identity function (a residual block), the zeros are kept, also for channels added by widening. Otherwise the normal initialization is used. If the example output is close to 0, a positive probe input decides.
- During tracing `active` is at its max, so the analysis sees all extra blocks. A copy of the original model is kept for `export()`.

**Why `tie_channels`.** During tracing, the layer after the block reads the output of the last extra block; in base mode it reads the block itself. A non-residual block has no addition to merge the two into one group, so sorting changes the network function without an error. `tie_channels(x, ref)` returns `x` at run time, but `torch.fx.wrap` leaves a node in the trace, and the analysis merges the two groups there.

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

**State, BN and export.**

- Width and depth form one state `(sizes, depths, mode)`. `mode_ctx`, `export` and `recalibrate_bn` restore it when they finish.
- BN updates its statistics only when both width and depth are base. So call `recalibrate_bn(..., depths=...)` before evaluating a deeper network.
- `set_random()` samples the widths first, then a uniform `0..k` for each depth point.
- `export()` renames `P.block.*` back to `P.*` and drops `P.extra.*`. `export(sizes, depths)` replaces each `DepthStack` with `nn.Sequential(block, *active extra blocks)`.

## 8. One training step

`netaug_step()` implements the paper's objective `L(base) + α × L(aug)`. The two gradients add up on the shared weights. The caller runs `optimizer.step()`:

```mermaid
flowchart LR
    A["set_base()"] --> B["forward<br/>base loss"]
    B --> C["backward"]
    C --> D["set_random()"]
    D --> E["forward<br/>augmented loss"]
    E --> F["backward<br/>α × augmented loss"]
    F --> G["set_base()"]
```

`recalibrate_bn()` resets the BN running statistics at the given width and depth, and runs the given batches with a cumulative average (`momentum=None`). When it finishes (also on error), it restores the state, the momentum and the per-module training flags. Extra blocks that do not run are not reset.

**The gradient on the base weights.** Take a two-layer linear network `y = V·W·x`. The augmented network adds `V_i·W_i·x`. With squared error:

```text
g_base = Vᵀ · r · xᵀ
g_aug  = g_base + Vᵀ · (V_i·W_i·x) · xᵀ
NetAug (α = 1): 2·g_base + Vᵀ · (V_i·W_i·x) · xᵀ
```

- The first term is close to doubling the learning rate of the base model. The second term is the signal specific to NetAug.
- In a nonlinear network, the extra channels also change the ReLU on/off pattern and the BN batch statistics.
- This is the structure of the gradient, not a proof of an effect. For the measured cosine and length ratio, see [Section 4.4 of experiments.en.md](experiments.en.md#44-gradients).

## 9. Correctness verification

"Correct" means that the package computes what the paper defines. Each link is compared with **plain models built directly in PyTorch, without this package**:

```mermaid
flowchart TD
    P["Paper: L(base) + α·L(aug), shared weights"]
    A["① base mode = original model"]
    B["② each width = plain model of that width"]
    C["③ gradient of one step = paper Eq. 2"]
    D["④ sorting keeps the function of the max-width network"]
    E["⑤ exported model = dynamic model"]
    P --> A & B
    A & B --> C
    C & D --> T["optimizer.step() receives the paper's update"]
    T --> E --> Z["deployed model = trained base model"]
```

`uv run pytest` runs 609 tests. They compare in float64 with tight tolerances. They randomize all parameters and BN statistics first, so default values cannot hide errors.

| Test file | Count | What it checks |
|---|---|---|
| `test_correctness.py` | 111 | ① base = original model; ⑤ export (dtype, precision, per-module flags); ④ sorting keeps the function of the max-width network, and is ordered and idempotent; optimizer state permutation; augmented networks do not pollute BN; ③ the gradient equals Eq. 2 element-wise; whitelist edge cases |
| `test_random_architectures.py` | 61 | ①④⑤ on 60 random architectures (residual, concat, depthwise, SE, and unsupported operations) |
| `test_whitelist_lemmas.py` | 172 | Lemma S and the weight lemma of the dynamic layers |
| `test_sorting.py` | 105 | Lemma I (60 random architectures); importance equals the official value on MBv2; the fallback and the extra-block rule; Theorem 3 and `.grad` permutation; error checks of `apply_permutations` |
| `test_widths.py` | 55 | ② parameter count, weight shapes, outputs and BN updates at each width equal a model of that width built by its own constructor |
| `test_depth.py` | 48 | ①③④⑤ for depth augmentation, identity initialization, state restore and error messages |
| `test_onnx.py` | 12 | Exported models converted to ONNX with dynamo and TorchScript; onnxruntime outputs match (difference < 2e-8) |
| `test_group_kinds.py` | 31 | Group kind counts of 4 torchvision models; block detection and `blocks=`; keys and errors of `scope`; sorting one kind keeps the function of the max-width network |
| `test_dynamic_usage.py` | 8 | Manual use of dynamic layers; rejection of models that contain dynamic layers |
| `test_docs.py` | 6 | The documentation examples run |

Mutation tests show that these tests are effective. The matching tests fail for each of these errors: `netaug_step` ignores `alpha`, uses a different augmented network, permutes only the producer outputs, or sorts in increasing or random order. Order after sorting is checked with a relative tolerance (1e-9 in float64, 1e-5 in float32), because the permutations of other groups change the summation order.

This chain shows that the code computes the paper's definition. It does **not** show that NetAug improves accuracy. That is an experimental question.

## 10. Differences from the official implementation

The table below is a summary. For the 20-item comparison, the nature of each difference and line links, see [official_diff.en.md](official_diff.en.md).

| Item | Official implementation | `torchnetaug` |
|---|---|---|
| Models | Hand-written MobileNet family | Any model that `torch.fx` can trace |
| Augmented dimensions | Width + expand ratio | Width of each channel group; optional depth |
| Width sampling | First block of each stage, stem and head each sample; expand per block | Once per group, or once for the whole model |
| Sorting | Only `inner` groups (expand and SE hidden channels) | All free groups by default, range chosen with `scope=`; one permutation per group, including optimizer state and `.grad` |
| BN in augmented networks | Updates statistics; reset before eval | No update by default; `recalibrate_bn` provided |
| Export | Hand-written `export()` | A copy of the original class |

## 11. Design decisions and limitations

**Plain layers only.** A model that contains a `DynamicModule` raises `TypeError`. The reasons: tracing treats dynamic layers as unknown operations; with a manual `graph=` they are not widened; and `export()` would depend on this package. An alternative that was not adopted is a set of package composition modules (`Sequential`, `Residual`, `Concat`) that declare the channel relations. It would support untraceable models, but it largely duplicates the automatic analysis. A custom tracer or a manual `ChannelGraph` already handles those cases.

| Limitation | Description |
|---|---|
| Supported layers | Only `nn.Conv2d`, `nn.Linear` (2D input) and BatchNorm are widened; the model must be traceable by `torch.fx` |
| Memory | The max-width network has about `max(width_mults)²` times the parameters of the base model |
| Weights | Layers with tied weights cannot be widened and raise `ValueError`; custom buffers registered on widened layers are lost |
| Depth augmentation | Only for blocks that keep the shape and run once per forward; cannot be combined with a manual `graph=` |
| Multi-GPU | DDP has not been tested |
