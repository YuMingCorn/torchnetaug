#!/usr/bin/env bash
# CIFAR-10, 3 seeds each:
#   1. baseline (TinyResNet x1), NetAug, NetAug + channel sorting every 10 epochs;
#   2. reference: plain TinyResNet trained alone at x1.5, x2, x2.5, x3, to compare
#      against the same widths exported from the NetAug model.
# Results are appended to runs/cifar10_compare.jsonl; finished runs are skipped, so
# the script can be restarted. Summarize with:
#   uv run python examples/summarize.py runs/cifar10_compare.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-40}
MODEL=${MODEL:-resnet_tiny}
OUT=${OUT:-runs/cifar10_compare.jsonl}
COMMON=(--skip-existing --dataset cifar10 --model "$MODEL" --epochs "$EPOCHS" --out "$OUT")

for seed in 0 1 2; do
  uv run python examples/cifar_netaug.py --method baseline --seed "$seed" "${COMMON[@]}"
  uv run python examples/cifar_netaug.py --method netaug   --seed "$seed" "${COMMON[@]}"
  uv run python examples/cifar_netaug.py --method netaug   --seed "$seed" --sort-every 10 "${COMMON[@]}"
done
for seed in 0 1 2; do
  for w in 1.5 2.0 2.5 3.0; do
    uv run python examples/cifar_netaug.py --method baseline --width "$w" --seed "$seed" "${COMMON[@]}"
  done
done
uv run python examples/summarize.py "$OUT"
