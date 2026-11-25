# Project summary: torchnetaug

[繁體中文](summary.md) | **English**

Status: finished (2026-11-24).

## Key points

1. **The implementation matches the paper.** 471 tests show that the package computes the values that the paper defines ([design.en.md Section 9](docs/design.en.md#9-correctness-verification)).
2. **NetAug did not raise the test accuracy of the base model.** On CIFAR-10 (3 seeds) and CIFAR-100 (8 seeds), every setting is within noise, gives no real gain, or is inconclusive.
3. **A baseline trained for 40 more epochs does better**: +1.54 percentage points, and the gain is significant. It still uses less compute than NetAug (in this environment, one NetAug step takes about 2.6 times as long as a baseline step).
4. **The extra loss signal did not turn into generalization.** The training accuracy went up (+1.16 percentage points on average at CIFAR-100 ×1), but the test accuracy did not. The augmented gradient points mostly in the same direction as the base gradient (cosine 0.82 to 0.93), so NetAug here acts close to a larger step size. The control for this was inconclusive, so it is only a conjecture.
5. **The conclusion applies only to the scope of this project**: CIFAR, TinyResNet, 40 epochs. The gains of the paper come from ImageNet, MobileNet-class models and 150 epochs. This project did not reproduce that setup.

## Main results

Paired difference in test accuracy of the base model against the baseline with the same seed (percentage points). For all results, see [experiments.en.md](docs/experiments.en.md).

| Setting | Difference | Reading |
|---|---|---|
| CIFAR-10, width augmentation (3 seeds) | +0.15 ± 0.74 | Within noise |
| CIFAR-10, depth augmentation (3 seeds) | −0.05 ± 0.29 | Within noise |
| CIFAR-100 ×0.25, NetAug (8 seeds) | −0.14 [−0.76, +0.49] | No real gain |
| CIFAR-100 ×0.5, NetAug (8 seeds) | +0.46 [−0.61, +1.52] | Inconclusive |
| CIFAR-100 ×1, NetAug (8 seeds) | −0.10 [−0.90, +0.69] | Inconclusive |
| CIFAR-100 ×0.25, baseline with 40 more epochs (8 seeds) | **+1.54 [+0.85, +2.24]** | Gain |

## Future work

- **Reproduce the paper's setup**: ImageNet, MobileNetV2-Tiny, 150 epochs with 5 epochs of warmup. If only 32×32 data is used, first change the strides of MobileNet.
- **Find why MobileNetV2 gets worse**: the cosine between its augmented gradient and its base gradient is only 0.12 to 0.64.
- **Narrow the confidence intervals of the remaining "inconclusive" settings**: 8 seeds resolved second-half stop, per-epoch sorting and global ×1/×2 to "no real gain", but `--init max`, ×0.5, ×1 and the paper's weight decay still span 0 and the +0.5 threshold; also design a control for the wide-path check in advance.
- **Lower the cost per step**: in this environment one NetAug step takes about 2.6 times the baseline step; the paper reports +16.7%.
- **Remove package limitations**: test DDP; keep custom buffers on widened layers; support `Conv1d`, `Conv3d` and transposed convolutions; support models that cannot be traced (a composite-module design could help; see [design.en.md Section 11](docs/design.en.md#11-design-decisions-and-limitations)).

## Documents

| Document | Content |
|---|---|
| [README.en.md](README.en.md) | Installation and API usage |
| [docs/design.en.md](docs/design.en.md) | Code design and correctness verification |
| [docs/experiments.en.md](docs/experiments.en.md) | Experiment setup, results and reproduction commands |
| [docs/glossary.md](docs/glossary.md) | Chinese–English term list |
| [docs/reference/](docs/reference/) | Notes on the paper and the official implementation |
