#!/usr/bin/env bash
# CIFAR-100 (a more under-fitted setting for TinyResNet than CIFAR-10), 3 seeds each:
#   1. baseline, lr 0.1
#   2. baseline, lr 0.2: control for "NetAug only doubles the base gradient"
#   3. NetAug, width multipliers 1 / 2 / 3
#   4. NetAug, width multipliers 1 / 1.2 / 1.4 (smaller augmentation)
# NetAug runs log the cosine and norm ratio of the base and augmented gradients on the
# base weights. Results are appended to runs/cifar100_regime.jsonl; finished runs are
# skipped, so the script can be restarted. Summarize with:
#   uv run python examples/summarize.py runs/cifar100_regime.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-40}
OUT=${OUT:-runs/cifar100_regime.jsonl}
COMMON=(--skip-existing --dataset cifar100 --model resnet_tiny --epochs "$EPOCHS" --out "$OUT")
GRAD=(--grad-log-every 10)

for seed in 0 1 2; do
  uv run python examples/cifar_netaug.py --method baseline --seed "$seed" "${COMMON[@]}"
  uv run python examples/cifar_netaug.py --method baseline --lr 0.2 --seed "$seed" "${COMMON[@]}"
  uv run python examples/cifar_netaug.py --method netaug --seed "$seed" "${GRAD[@]}" "${COMMON[@]}"
  uv run python examples/cifar_netaug.py --method netaug --width-mults 1.0 1.2 1.4 \
    --eval-widths 1.0 1.2 1.4 --seed "$seed" "${GRAD[@]}" "${COMMON[@]}"
done
uv run python examples/summarize.py "$OUT"
