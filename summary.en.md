# Project summary: torchnetaug

[繁體中文](summary.md) | **English**

Status: finished (2026-11-23).

## Conclusion

This project implemented Network Augmentation (NetAug, Cai et al., ICLR 2022) in pure PyTorch and tested its effect on CIFAR-10 and CIFAR-100. **The implementation is correct, but NetAug gave no significant gain in any tested setting.** The extra loss signal from the augmented networks often raised the training accuracy, but it did not raise the test accuracy of the base model. The same compute spent on 40 more epochs of training gave a significantly better result.

## Work done

- **The `torchnetaug` package** (`src/torchnetaug/`, MIT license): it wraps any model that `torch.fx` can trace and that is built from plain `torch.nn` layers. It analyzes the channel structure automatically; it does width and depth augmentation; it sorts channels per channel group, which keeps the max-width function on residual and concat networks; and its export returns the original model class, which can be converted to ONNX.
- **Correctness**: 471 tests show that each width equals a plain model of that width, that the gradient of one training step equals Eq. 2 of the paper, that sorting does not change the max-width function, and that the exported model equals the dynamic network. See [design.en.md Section 9](docs/design.en.md#9-correctness-verification).
- **Experiments**: width, depth and sorting on CIFAR-10; a diagnosis of an underfitting CIFAR-100 setup with controls and the recipe options of the official code; a first test of MobileNetV2. See [experiments.en.md](docs/experiments.en.md).

## Main results

Paired difference in test accuracy of the base model against the baseline with the same seed (percentage points):

| Setting | Difference | Reading |
|---|---|---|
| CIFAR-10, width augmentation (3 seeds) | +0.15 ± 0.74 | Within noise |
| CIFAR-10, width augmentation + sorting (3 seeds) | +0.34 ± 0.44 | Within noise |
| CIFAR-10, depth augmentation (3 seeds) | −0.05 ± 0.29 | Within noise |
| CIFAR-100 ×0.25, NetAug (8 seeds) | −0.14 [−0.76, +0.49] | No real gain |
| CIFAR-100 ×1, NetAug (5 seeds) | −0.46 [−1.05, +0.14] | No real gain |
| CIFAR-100 ×0.25, official recipe options and the paper's weight decay (5 seeds each) | −0.25 to +0.32, all intervals contain 0 | Inconclusive |
| CIFAR-100 ×0.25, baseline with 40 more epochs (8 seeds) | **+1.54 [+0.85, +2.24]** | Gain |
| CIFAR-100 MobileNetV2 0.35 + warmup (1 seed) | −4.06 | Worse, cause not known |

## Observations

1. **The extra loss signal did not turn into generalization.** The training accuracy went up (+0.7 to +1.3 percentage points per seed on CIFAR-10); the test accuracy did not.
2. **The augmented gradient points mostly in the same direction as the base gradient** (cosine 0.82 to 0.93 on TinyResNet), so NetAug here acts almost like a larger step size. But the control for "only a larger step size" is "inconclusive", so this is a guess.
3. **Sorting almost never changes the base.** The added channels start with a smaller initialization, and the base channels get a gradient at every step, so the base channels stay the most important.
4. **An exported wide model cannot replace a wide model trained alone** (1.3 to 15 percentage points lower).
5. **The experiments differ from the paper.** The gains of the paper come from ImageNet, MobileNet-class models and 150 epochs. This project did not reproduce that setup, so the conclusion applies only to small datasets, short training and TinyResNet.

## Future work

- **Reproduce the paper's setup**: ImageNet, MobileNetV2-Tiny, 150 epochs with 5 epochs of warmup. If only 32×32 data is used, first change the strides of MobileNet.
- **Find why MobileNetV2 gets worse**: the cosine between its augmented gradient and its base gradient is only 0.12 to 0.64.
- **Add seeds for the "inconclusive" settings**: `--init max`, `--stop-aug`, ×0.5, the paper's weight decay; and design a control for the wide-path check in advance.
- **Lower the cost per step**: in this environment one NetAug step takes about 2.6 times the baseline step; the paper reports +16.7%.
- **Package limitations**: test DDP; custom buffers on widened layers; `Conv1d`, `Conv3d`, transposed convolution; models that cannot be traced (the composition-module alternative could be evaluated, see [design.en.md Section 11](docs/design.en.md#11-design-decisions-and-limitations)).

## Documents

| Document | Content |
|---|---|
| [README.en.md](README.en.md) | Installation and API usage |
| [docs/design.en.md](docs/design.en.md) | Code design and correctness verification |
| [docs/experiments.en.md](docs/experiments.en.md) | Experiment setup, results, observations and reproduction commands |
| [docs/glossary.md](docs/glossary.md) | Chinese–English term list |
| [docs/reference/](docs/reference/) | Notes on the paper, the official code and the reference implementation |
