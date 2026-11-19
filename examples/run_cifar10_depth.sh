#!/usr/bin/env bash
# CIFAR-10 depth augmentation, 3 seeds each (TinyResNet, same recipe as run_cifar10_compare.sh):
#   1. depth only:     one extra block after each stage, width not augmented;
#   2. width + depth:  widths x1/x2/x3 and one extra block after each stage.
# Pair the results by seed with the baseline / netaug runs in runs/cifar10_compare.jsonl
# (same base initialization and data order for the same seed).
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-40}
OUT=${OUT:-runs/cifar10_depth.jsonl}
COMMON=(--skip-existing --dataset cifar10 --model resnet_tiny --epochs "$EPOCHS" --out "$OUT" --method netaug)

for seed in 0 1 2; do
  uv run python examples/cifar_netaug.py --seed "$seed" --depth 1 --width-mults 1.0 --eval-widths 1.0 "${COMMON[@]}"
  uv run python examples/cifar_netaug.py --seed "$seed" --depth 1 "${COMMON[@]}"
done
uv run python examples/summarize.py "$OUT"
