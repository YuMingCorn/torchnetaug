"""Summarize a results .jsonl file written by examples/cifar_netaug.py.

    uv run python examples/summarize.py [results.jsonl]

Prints three tables:
1. mean ± std per configuration (dataset, model, epochs, method);
2. paired per-seed differences against the baseline with the same seed;
3. accuracy and parameter count at each evaluated width (NetAug runs only),
   plus the per-group random widths that training actually samples.
"""

import json
import statistics
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
path = sys.argv[1] if len(sys.argv) > 1 else "runs/cifar_results.jsonl"
groups = defaultdict(list)
with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r["method"] == "baseline":
            w = r.get("width", 1.0)
            label = "baseline" if w == 1.0 else f"baseline x{w}"
        else:
            label = f"netaug{'+sort' + str(r['sort_every']) if r['sort_every'] else ''}"
        key = (r["dataset"], r["model"], r["epochs"], label)
        groups[key].append(r)


def fmt(xs, scale=100.0):
    m = statistics.mean(xs) * scale
    s = statistics.stdev(xs) * scale if len(xs) > 1 else float("nan")
    return f"{m:.2f} ± {s:.2f}"


print("## Base-width accuracy\n")
print("| dataset | model | epochs | method | seeds | params | test acc (%) | train acc (%) | minutes/run |")
print("|---|---|---|---|---|---|---|---|---|")
for (ds, model, ep, label), rs in sorted(groups.items()):
    seeds = ",".join(str(r["seed"]) for r in rs)
    mins = statistics.mean(r["minutes"] for r in rs)
    print(f"| {ds} | {model} | {ep} | {label} | {seeds} | {rs[0]['params']:,} | "
          f"{fmt([r['test_acc'] for r in rs])} | {fmt([r['train_acc'] for r in rs])} | {mins:.1f} |")

# Paired differences: same seed means the same initialization of the base weights and
# the same data order at the start, so the per-seed difference removes part of the noise.
print("\n## Paired difference against the baseline (test acc, percentage points)\n")
print("| dataset | model | epochs | method | per-seed diff | mean ± std | baseline std |")
print("|---|---|---|---|---|---|---|")
for (ds, model, ep, label), rs in sorted(groups.items()):
    if label.startswith("baseline"):
        continue
    base = {r["seed"]: r["test_acc"] for r in groups.get((ds, model, ep, "baseline"), [])}
    pairs = [(r["seed"], r["test_acc"] - base[r["seed"]]) for r in sorted(rs, key=lambda r: r["seed"]) if r["seed"] in base]
    if not pairs:
        continue
    diffs = [d for _, d in pairs]
    per_seed = ", ".join(f"s{s}: {d * 100:+.2f}" for s, d in pairs)
    base_runs = [v for v in base.values()]
    base_std = f"{statistics.stdev(base_runs) * 100:.2f}" if len(base_runs) > 1 else "n/a"
    print(f"| {ds} | {model} | {ep} | {label} | {per_seed} | {fmt(diffs)} | {base_std} |")

# Accuracy vs. width: every width exported from NetAug as a standalone plain-PyTorch
# model, next to the same architecture trained alone at that width.
widths = sorted({w for rs in groups.values() for r in rs for w in r.get("width_acc", {})}, key=float)
for ds, model, ep in sorted({k[:3] for k in groups}):
    netaug_labels = [k[3] for k in sorted(groups) if k[:3] == (ds, model, ep) and k[3].startswith("netaug")]
    if not widths or not netaug_labels:
        continue
    print(f"\n## Exported models at each width ({ds}, {model}, {ep} epochs)\n")
    print("Test accuracy (%). Each NetAug width is BN-recalibrated, exported with `export(sizes)` "
          "(plain PyTorch, no torchnetaug layers) and evaluated. \"trained alone\" is the same "
          "architecture trained by itself at that width with the baseline recipe.\n")
    head = ["width", "params", "trained alone"] + [f"{l} exported" for l in netaug_labels] + ["max abs(exported - dynamic)"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for w in widths:
        alone_label = "baseline" if float(w) == 1.0 else f"baseline x{float(w)}"
        alone = groups.get((ds, model, ep, alone_label), [])
        cells = [f"x{w}"]
        any_run = next(r for l in netaug_labels for r in groups[(ds, model, ep, l)] if r.get("width_acc"))
        cells.append(f"{any_run['width_acc'][w]['params']:,}")
        cells.append(fmt([r["test_acc"] for r in alone]) + f" (n={len(alone)})" if alone else "—")
        diffs = []
        for l in netaug_labels:
            rs = [r for r in groups[(ds, model, ep, l)] if r.get("width_acc")]
            key = "export_test_acc" if all("export_test_acc" in r["width_acc"][w] for r in rs) else "test_acc"
            cells.append(fmt([r["width_acc"][w][key] for r in rs]) + f" (n={len(rs)})")
            diffs += [r["width_acc"][w].get("export_max_abs_diff", float("nan")) for r in rs]
        cells.append(f"{max(diffs):.1e}")
        print("| " + " | ".join(cells) + " |")

rand = [(k, rs) for k, rs in sorted(groups.items()) if any(r.get("random_acc") for r in rs)]
if rand:
    print("\n## Per-group random widths (the distribution sampled in training)\n")
    print("| dataset | model | epochs | method | configs | params (min–max) | test acc (%) mean ± std | min | max |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (ds, model, ep, label), rs in rand:
        items = [x for r in rs for x in r.get("random_acc", [])]
        accs = [x["test_acc"] for x in items]
        ps = [x["params"] for x in items]
        print(f"| {ds} | {model} | {ep} | {label} | {len(items)} | {min(ps):,}–{max(ps):,} | "
              f"{fmt(accs)} | {min(accs) * 100:.2f} | {max(accs) * 100:.2f} |")
