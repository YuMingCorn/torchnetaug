#!/usr/bin/env bash
# Diagnosis of "NetAug shows no clear gain": is the setting under-fitted, and does the
# training recipe matter? Every run writes its own file runs/diag/<name>.jsonl (and .log);
# finished runs are skipped, so the script can be restarted.
#
#   STAGES="main"                       bash examples/run_diagnosis.sh
#   STAGES="dose controls ablations"    bash examples/run_diagnosis.sh
#
# Stages (TinyResNet, 40 epochs unless noted):
#   main       baseline and NetAug at the most under-fitted setting ($DATASET, base width $WIDTH)
#   dose       baseline and NetAug at the other base widths ($DOSE_WIDTHS): does a narrower
#              (more under-fitted) base model gain more?
#   controls   at the main setting: baseline with lr 0.2 and weight decay 2.5e-4 (the same
#              update as a doubled loss: "NetAug only doubles the gradient"), and baseline
#              with twice the epochs (about the same compute as NetAug)
#   alone      plain models trained alone at $ALONE_WIDTHS (the exported x2 and x3 of the
#              main base width), to compare with the wider models exported from NetAug
#   ablations  at the main setting, recipe choices of the official code: no augmentation
#              in the second half; the initializer applied to the whole max-width network,
#              with and without sorting the channels every epoch
#   sanity     NetAug with width multipliers 1 and 2 sampled for the whole network: half of
#              the augmented steps train the full x2 network. Is the exported x2 model close to
#              the x2 model trained alone? (checks the wide path of the wrapper)
#   recipe     baseline and NetAug with the paper's weight decay (5e-5 here, 4e-5 in the paper)
#
# Summarize with:
#   uv run python examples/summarize.py runs/diag/*.jsonl
#   uv run python examples/summarize.py --ref-epochs 40 runs/diag/*.jsonl   # 80-epoch control
set -uo pipefail
cd "$(dirname "$0")/.."

STAGES=${STAGES:-main}
DATASET=${DATASET:-cifar100}
WIDTH=${WIDTH:-0.25}
DOSE_WIDTHS=${DOSE_WIDTHS:-"0.5 1.0"}
ALONE_WIDTHS=${ALONE_WIDTHS:-"0.5 0.75"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
ABLATION_SEEDS=${ABLATION_SEEDS:-"0 1 2 3 4"}
EPOCHS=${EPOCHS:-40}
OUT=${OUT:-runs/diag}
mkdir -p "$OUT"

failed=0
run() {  # arguments of cifar_netaug.py; the file name is made from them
  local name
  name=$(echo "$*" | sed -e 's/--//g' -e 's/ /_/g')
  if ! uv run python examples/cifar_netaug.py --skip-existing --model resnet_tiny "$@" \
      --out "$OUT/$name.jsonl" > "$OUT/$name.log" 2>&1; then
    echo "FAILED: $*  (see $OUT/$name.log)"
    failed=1
  else
    echo "done: $*"
  fi
}

pair() {  # baseline and NetAug with the same seed; other arguments are shared
  run --method baseline "$@"
  run --method netaug --grad-log-every 20 --eval-widths 1.0 2.0 3.0 "$@"
}

for stage in $STAGES; do
  case "$stage" in
    main)
      for seed in $SEEDS; do
        pair --dataset "$DATASET" --width "$WIDTH" --epochs "$EPOCHS" --seed "$seed"
      done ;;
    dose)
      for seed in $SEEDS; do
        for w in $DOSE_WIDTHS; do
          pair --dataset "$DATASET" --width "$w" --epochs "$EPOCHS" --seed "$seed"
        done
      done ;;
    alone)  # plain models at the widths the NetAug models export (x2 and x3 of the base)
      for seed in $ABLATION_SEEDS; do
        for w in $ALONE_WIDTHS; do
          run --method baseline --dataset "$DATASET" --width "$w" --epochs "$EPOCHS" --seed "$seed"
        done
      done ;;
    controls)
      for seed in $SEEDS; do
        run --method baseline --lr 0.2 --wd 2.5e-4 --dataset "$DATASET" --width "$WIDTH" --epochs "$EPOCHS" --seed "$seed"
        run --method baseline --dataset "$DATASET" --width "$WIDTH" --epochs $((2 * EPOCHS)) --seed "$seed"
      done ;;
    ablations)
      for seed in $ABLATION_SEEDS; do
        common=(--method netaug --eval-widths 1.0 2.0 3.0 --dataset "$DATASET" --width "$WIDTH" --epochs "$EPOCHS" --seed "$seed")
        run "${common[@]}" --stop-aug 0.5
        run "${common[@]}" --init max
        run "${common[@]}" --init max --sort-every 1
      done ;;
    sanity)
      for seed in 0 1 2; do
        run --method netaug --width-mults 1.0 2.0 --granularity global --eval-widths 1.0 2.0             --dataset "$DATASET" --width "$WIDTH" --epochs "$EPOCHS" --seed "$seed"
      done ;;
    recipe)
      for seed in $ABLATION_SEEDS; do
        run --method baseline --wd 5e-5 --dataset "$DATASET" --width "$WIDTH" --epochs "$EPOCHS" --seed "$seed"
        run --method netaug --wd 5e-5 --eval-widths 1.0 2.0 3.0 --dataset "$DATASET" --width "$WIDTH" --epochs "$EPOCHS" --seed "$seed"
      done ;;
    *)
      echo "unknown stage: $stage"; exit 2 ;;
  esac
done
exit $failed
