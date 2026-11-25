"""Channel-sorting ablation on CIFAR-100: one factor changes at a time.

    uv run python examples/sort_ablation.py [--list] [--smoke] [--seeds 0 1 2]

Each run is one call of examples/cifar_netaug.py with ``--skip-existing``, so the script
can be stopped and started again: finished runs are skipped.

Settings (the reference is ``all`` = every free group, summed importance, permuted
optimizer state):

* scope:      none (no sorting), inner, closed, all
* importance: all with ``--sort-agg mean``, all with ``--sort-agg single``
* optimizer:  all with ``--sort-opt keep``, all with ``--sort-opt rebuild``

``closed`` is skipped for a model without closed groups (resnet_tiny), because it would
repeat ``inner``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SETTINGS: dict[str, list[str]] = {
    "none": ["--sort-every", "0"],
    "inner": ["--sort-every", "1", "--sort-scope", "inner"],
    "closed": ["--sort-every", "1", "--sort-scope", "closed"],
    "all": ["--sort-every", "1"],
    "all-mean": ["--sort-every", "1", "--sort-agg", "mean"],
    "all-single": ["--sort-every", "1", "--sort-agg", "single"],
    "all-keep": ["--sort-every", "1", "--sort-opt", "keep"],
    "all-rebuild": ["--sort-every", "1", "--sort-opt", "rebuild"],
}
NO_CLOSED_GROUPS = {"resnet_tiny"}


def jobs(models: list[str], seeds: list[int], settings: list[str]) -> list[tuple[str, int, str]]:
    return [
        (model, seed, setting)
        for seed in seeds
        for model in models
        for setting in settings
        if not (setting == "closed" and model in NO_CLOSED_GROUPS)
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["mbv2_035", "resnet_tiny"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--amp", default="fp16", choices=["bf16", "fp16", "off"], help="fp16: the Colab T4 has no fast bf16")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/sort_ablation.jsonl")
    p.add_argument("--list", action="store_true", help="print the runs and exit")
    p.add_argument("--smoke", action="store_true", help="2 epochs, seed 0, results in <out>.smoke.jsonl")
    args = p.parse_args()

    out = Path(args.out)
    if args.smoke:
        args.epochs, args.seeds = 2, [0]
        out = out.with_suffix(".smoke.jsonl")
    todo = jobs(args.models, args.seeds, args.settings)
    if args.list:
        for model, seed, setting in todo:
            print(model, seed, setting)
        print(len(todo), "runs")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("cifar_netaug.py")
    failed = []
    for i, (model, seed, setting) in enumerate(todo, 1):
        cmd = [sys.executable, str(script), "--method", "netaug", "--dataset", "cifar100"]
        cmd += ["--model", model, "--seed", str(seed), "--epochs", str(args.epochs), "--amp", args.amp]
        cmd += ["--data", args.data, "--out", str(out), "--skip-existing", *SETTINGS[setting]]
        start = time.time()
        code = subprocess.run(cmd, check=False).returncode
        status = "ok" if code == 0 else f"FAILED ({code})"
        print(
            f"[{i}/{len(todo)}] {model} seed {seed} {setting}: {status} {(time.time() - start) / 60:.1f} min",
            flush=True,
        )
        if code != 0:
            failed.append((model, seed, setting))
    print("ALL_RUNS_FINISHED" if not failed else f"FAILED_RUNS {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
