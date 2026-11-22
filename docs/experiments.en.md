# torchnetaug experiment results and observations

[繁體中文](experiments.md) | **English**

Dates: 2026-11-19 (CIFAR-10), 2026-11-23 (CIFAR-100). For the code design and correctness verification, see [design.en.md](design.en.md). For the project summary, see [summary.en.md](../summary.en.md). For terms, see [glossary.md](glossary.md).

## 1. Conclusions

- **In every tested setting, NetAug did not significantly improve the test accuracy of the base model.** On CIFAR-10 the gain is at most +0.34 percentage points, smaller than the seed noise. On CIFAR-100, no NetAug setting has a 95% confidence interval with a lower bound above 0.
- **The same compute spent on 40 more epochs of training gives +1.54 percentage points (95% CI [+0.85, +2.24]), and this gain is significant.**
- The training accuracy often goes up (+0.7 to +1.3 percentage points per seed on CIFAR-10), but this does not turn into test accuracy.
- Depth augmentation, channel sorting, and the recipe options of the official code did not change the result.
- Wide models exported from a NetAug model are 1.3 to 15 percentage points below models of the same width trained alone.
- MobileNetV2 (1 seed) trains worse with NetAug. The cause is not known.
- No implementation error that could make NetAug fail was found. For all exported models, the output difference from the dynamic network is 0.

## 2. Common setup

| Item | Setting |
|---|---|
| Code | Branch `feat/torchnetaug`: CIFAR-10 width `941b37a`, depth `83e5073`, CIFAR-100 `f81bdfc` |
| Environment | Python 3.14.3, PyTorch 2.11.0+cu128, Windows 10, RTX 3060 Ti 8 GB |
| Model | TinyResNet: CIFAR ResNet-18 at 1/8 width (8, 16, 32, 64 channels, 176,402 parameters); `--width` multiplies this by a base width multiplier |
| Training | 40 epochs, batch 128, Nesterov SGD (momentum 0.9), lr 0.1 cosine, weight decay 5e-4 (not on BN and bias), random crop and horizontal flip, bf16 autocast |
| NetAug | `width_mults=(1.0, 2.0, 3.0)`, `alpha=1.0`, each channel group samples its own width, augmentation for the whole run |
| Evaluation | Before evaluation, the BN statistics of each model are re-estimated on 10,000 training images (no data augmentation). Test accuracy uses the 10,000 test images; training accuracy uses the first 10,000 training images |
| Comparison | Runs with the same seed have the same base initialization and data order, so each run is paired with the baseline of the same seed and subtracted |

## 3. CIFAR-10

### 3.1 Width augmentation

3 seeds per setting. `+sort10` means the channels are sorted every 10 epochs.

| Method | test acc (%) | train acc (%) | Paired difference (points) |
|---|---|---|---|
| baseline | 86.92 ± 0.35 | 91.90 ± 0.23 | — |
| netaug | 87.07 ± 0.39 | 92.70 ± 0.20 | +0.15 ± 0.74 (s0 +0.83, s1 +0.27, s2 −0.64) |
| netaug+sort10 | 87.26 ± 0.09 | 92.81 ± 0.23 | +0.34 ± 0.44 (s0 +0.76, s1 +0.37, s2 −0.11) |

Each width exported from the NetAug model, compared with a model of the same width trained alone (test acc %):

| Width | Parameters | Trained alone | netaug export | netaug+sort10 export |
|---|---|---|---|---|
| ×1.0 | 176,402 | 86.92 ± 0.35 | 87.07 ± 0.39 | 87.26 ± 0.09 |
| ×1.5 | 395,350 | 89.27 ± 0.11 | 87.48 ± 0.25 | 88.00 ± 0.13 |
| ×2.0 | 701,466 | 90.46 ± 0.27 | 88.22 ± 0.16 | 88.48 ± 0.21 |
| ×2.5 | 1,094,750 | 91.27 ± 0.33 | 88.32 ± 0.31 | 88.49 ± 0.23 |
| ×3.0 | 1,575,202 | 91.62 ± 0.30 | 88.44 ± 0.18 | 88.50 ± 0.27 |

Observations:

- The gain of the base is smaller than the standard deviation of the paired difference, and in each method one of the 3 seeds is negative.
- The training accuracy goes up by 0.7 to 1.3 percentage points for every seed. This agrees with the mechanism "reduce underfitting", but the test accuracy does not go up with it.
- From ×1.5 up, the exported wide models are 1.3 to 3.2 percentage points lower, and the gap grows with the width. The added channels get gradients only when they are sampled.
- The gain of the paper (ImageNet, MobileNetV2-Tiny, 150 epochs, +1.6 percentage points) was not reproduced. The dataset, the model and the training length are all different.

### 3.2 Channel sorting

- In 9 sorts (3 seeds × epochs 10, 20, 30), the largest change of the max-width output is 8.6e-6 (the logits are about 20 to 31). So sorting keeps the function on really trained weights.
- **None of the 9 sorts moved any channel into the base.** The added channels use the default initialization, and PyTorch scales it by the number of input channels, so the new weights are smaller (for example, in `layers.0.conv1` the standard deviation is about 0.070 in the base block and 0.039 in the new block). Before training, the smallest importance of the base channels in each group is already 1.06 to 1.19 times the largest importance of the new channels. During training the base channels get a gradient at every step, so the gap stays.
- Sorting only changes which channels the middle widths keep (before training, 168 to 188 of the 720 channels kept by ×2 are swapped). So `+sort10` helps the wider models a little (×1.5 +0.52, ×2.0 +0.26), and the difference at the base should be read as seed noise.

### 3.3 Depth augmentation

One extra block after the last block of each stage (`--depth 1`), 3 seeds per setting.

| Method | test acc (%) | train acc (%) | Paired difference (points) |
|---|---|---|---|
| baseline | 86.92 ± 0.35 | 91.90 ± 0.23 | — |
| Depth augmentation | 86.87 ± 0.20 | 92.28 ± 0.16 | −0.05 ± 0.29 |
| Width + depth augmentation | 86.96 ± 0.05 | 92.77 ± 0.21 | +0.04 ± 0.34 |

- Depth augmentation has no measurable effect on the base, and it does not make width augmentation better.
- The exported network with all extra blocks on has 56% more parameters (274,802 versus 176,402) but the same accuracy (86.92 ± 0.17). The extra blocks are residual copies that start as identity functions and are trained only when sampled, so they learn almost nothing in 40 epochs.

## 4. CIFAR-100 diagnosis

The effect on CIFAR-10 is small. This section checks why: the setup does not suit NetAug, the implementation has a problem, or NetAug has no effect in this kind of setup.

### 4.1 Design

- **An underfitting setup.** CIFAR-100, TinyResNet at base width ×0.25 (2, 4, 8, 16 channels), with a training accuracy of only 33.8%. ×0.5 and ×1 are also tested.
- **Reading rules fixed in advance.** For the 95% confidence interval (t distribution) of the paired difference: lower bound above 0 means "gain"; upper bound below +0.5 percentage points means "no real gain"; otherwise "inconclusive".
- **Controls.** lr 0.2 with weight decay 2.5e-4 (NetAug adds two losses, which is like a larger gradient); a baseline trained for 80 epochs (about the compute of NetAug); ×2 and ×3 models trained alone.

### 4.2 Base model

| Setting | n | test acc (%) | Test difference [95% CI] | Train difference [95% CI] | Reading |
|---|---|---|---|---|---|
| ×0.25 baseline | 8 | 32.54 | — | — | — |
| ×0.25 NetAug | 8 | 32.41 | −0.14 [−0.76, +0.49] | +0.08 [−0.43, +0.58] | No real gain |
| ×0.25 NetAug, initialize the whole max-width network (`--init max`) | 5 | 32.72 | +0.32 [−1.11, +1.74] | +0.04 [−1.24, +1.32] | Inconclusive |
| ×0.25 NetAug, stop augmentation for the second half (`--stop-aug 0.5`) | 5 | 32.59 | +0.18 [−0.65, +1.00] | +0.16 [−0.38, +0.69] | Inconclusive |
| ×0.25 NetAug, sort every epoch + `--init max` | 5 | 32.16 | −0.25 [−1.44, +0.94] | −0.38 [−2.11, +1.35] | Inconclusive |
| ×0.25 NetAug, weight decay 5e-5 (against the baseline with the same weight decay) | 5 | 30.81 | −0.22 [−2.34, +1.89] | −0.22 [−2.50, +2.07] | Inconclusive |
| ×0.5 NetAug | 5 | 47.31 | +0.59 [−1.19, +2.36] | +0.66 [−0.95, +2.27] | Inconclusive |
| ×1 NetAug | 5 | 59.52 | −0.46 [−1.05, +0.14] | +0.89 [−0.40, +2.18] | No real gain |
| Control: ×0.25 baseline, lr 0.2, weight decay 2.5e-4 | 8 | 32.48 | −0.06 [−1.17, +1.04] | +0.02 [−1.34, +1.38] | Inconclusive |
| Control: ×0.25 baseline, weight decay 5e-5 | 5 | 31.03 | −1.38 [−2.78, +0.03] | −1.64 [−3.04, −0.24] | No real gain |
| Control: ×0.25 baseline, 80 epochs | 8 | 34.09 | **+1.54 [+0.85, +2.24]** | +1.58 [+0.97, +2.19] | Gain |

![Paired differences](figures/paired_diff.png)

- No NetAug setting gives a test gain. The upper bound at ×0.25, +0.49, is only 0.01 below the threshold, so "no real gain" only just holds.
- 40 more epochs of training give a significant gain. In this environment one NetAug step takes about 2.6 times the baseline step (the paper reports +16.7%), so this compute comparison applies only to this environment.
- The difference in training accuracy grows with the base width (×0.25 +0.08, ×0.5 +0.66, ×1 +0.89), but all three intervals contain 0. At ×1 the training accuracy goes up and the test accuracy goes down, the same as ResNet50 in the paper.
- The paper says smaller models gain more. Here the most underfitting model, ×0.25, gains the least.

### 4.3 Exported wide models

![Width multiplier](figures/acc_vs_width_mult.png)

![Parameter count](figures/acc_vs_params.png)

| Base width | Exported ×2 | Exported ×3 | Same width trained alone |
|---|---|---|---|
| ×0.25 | 39.04 | 40.47 | 46.72 (×2), 55.13 (×3) |
| ×0.25, stop augmentation 0.5 | 34.61 | 35.02 | Same as above |
| ×0.5 | 52.28 | 53.06 | 59.98 (×2) |

- The exported wide models are 7 to 15 percentage points below models of the same width trained alone. When each group samples its own width, the whole ×3 network is almost never sampled. The networks actually sampled in training (19,000 to 92,000 parameters) give 36.97 ± 2.41% on average after export, while a 47,600-parameter model trained alone gives 46.7%.
- **Wide-path check**: when the whole network samples ×1 or ×2 together, the exported ×2 gives 41.22 ± 1.56% (n=3). The ×2 network is trained in only about half of the steps, and the after-the-fact control, a ×2 model trained alone for 20 epochs, gives 43.2% (n=3), about 2 percentage points higher. This control was not designed in advance, so the reading is "inconclusive": no clear defect was seen, but the check does not prove that the wide path has no problem.

### 4.4 Gradients

![Gradients](figures/gradient.png)

Every 20 steps, the cosine and the length ratio between the augmented gradient and the base gradient on the base weights are logged. For all three widths the cosine is 0.82 to 0.93 and the length ratio is 0.87 to 0.97. At a cosine of 0.82 to 0.92, the component perpendicular to the base gradient is still about 40% to 57% of the base gradient length, so it is wrong to say that the augmented gradient carries no new information. The result agrees with "NetAug acts mostly like a larger step size here", but the lr 0.2 control that checks this idea is "inconclusive", so the idea is not proven.

### 4.5 Sorting

With a sort every epoch, the 39 sorts of each seed move only 0 to 4 channels into the base in total, which agrees with Section 3.2.

### 4.6 MobileNetV2 (preliminary, 1 seed)

The gains of the paper come from MobileNet-class models, so torchvision MobileNetV2 (width 0.35, 520,000 parameters) was also tested.

| Setting | baseline | NetAug |
|---|---|---|
| 40 epochs | 41.61% | 36.50% (exported ×2 38.3%, ×3 38.6%) |
| 40 epochs + 5 epochs warmup | 44.86% | 40.80% (exported ×2 42.27%, ×3 42.46%) |
| 10 epochs | 20.7% | 13.0% |

- The base loss of NetAug is higher than the baseline from the first epoch. In the first epoch the cosine between the augmented gradient and the base gradient is only 0.25 (length ratio 3.1); after that it is 0.4 to 0.64 (TinyResNet: 0.8 to 0.93).
- 10-epoch ablations: NetAug with lr 0.05 gives 18.2%, NetAug with α=0 gives 19.6%, and the baseline with lr 0.2 and weight decay 2.5e-4 gives 20.8%. So the forward pass of the wrapper is not the cause, and doubling the step size alone does not make the baseline worse; the problem is in the gradient of the augmented loss.
- A fan-out initialization of the added channels was worse (10 epochs: 10.0%) and was withdrawn. Warmup lowers the length ratio of the first epoch to 1.2, but the cosine is still only 0.12, and NetAug is still 4.1 percentage points lower.
- This model is not suitable for a direct conclusion: the total stride is 32, so a 32×32 input becomes 1×1 at the end, and the baseline reaches only 41.6%. Width still helps: width 0.5 trained alone gives 45.02% (816,000 parameters).
- The raw data is in `runs/diag/mbv2_default_init/`.

### 4.7 Possible causes (not proven)

The gains of the paper come from MobileNet-class models on ImageNet, where augmentation widens the expand channels of the inverted blocks, trained for 150 epochs with 5 epochs of warmup. The architecture, the dataset and the training length of these experiments are all different.

## 5. Limitations

- CIFAR-10 has only 3 seeds per setting; CIFAR-100 has 3 to 8 seeds per setting; MobileNetV2 has only 1 seed. Differences smaller than about 0.5 percentage points cannot be told apart from seed noise.
- Only 40 epochs (80 for a control), 32×32 images and two architectures were used. ImageNet was not tested.
- During some runs the GPU was also running other programs, so the run times are only approximate.

## 6. Reproduction

Put the data in `data/` (if the SSL certificate of the official CIFAR host has expired, download with `curl -L`). Each run appends one line of results to a jsonl file.

```bash
uv run python examples/check_widths_fake_data.py       # fake data: size and computation at each width are correct
bash examples/run_cifar10_compare.sh                   # Section 3.1
bash examples/run_cifar10_depth.sh                     # Section 3.3
uv run python examples/summarize.py runs/cifar10_compare.jsonl runs/cifar10_depth.jsonl

STAGES="main controls alone ablations" bash examples/run_diagnosis.sh   # Section 4
SEEDS="0 1 2 3 4" STAGES="dose" bash examples/run_diagnosis.sh
STAGES="sanity recipe" bash examples/run_diagnosis.sh
uv run python examples/summarize.py --ref-epochs 40 runs/diag/method_*.jsonl
uv run python examples/plot_results.py --out runs/plots runs/diag/method_*.jsonl
```

- The stages of `run_diagnosis.sh`: `main` (8 seeds), `dose` (base width 0.5, 1.0), `controls`, `alone`, `ablations`, `sanity`, `recipe`. Finished runs are skipped.
- Main options of `cifar_netaug.py`: `--width` (base width multiplier), `--depth`, `--init {base,max}`, `--stop-aug`, `--sort-every`, `--granularity {group,global}`, `--grad-log-every N`, `--amp {bf16,fp16,off}`, `--warmup N`.
- `examples/run_cifar100_regime.sh` is an earlier CIFAR-100 control script. This document does not use its results.
- MobileNetV2: `uv run python examples/cifar_netaug.py --model {mbv2_035,mbv2_050} --method {baseline,netaug} --dataset cifar100 --epochs 40 --seed 0 --eval-widths 1.0 2.0 3.0 --out <file>`, optionally with `--warmup 5`, `--grad-log-every 20`, `--alpha 0`, or `--lr 0.05 --wd 1e-3`.
