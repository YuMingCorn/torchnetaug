"""Summarize a results .jsonl file written by examples/cifar_netaug.py.

    uv run python examples/summarize.py [--ref-epochs N] [results.jsonl ...]

Several files are read together (e.g. runs/cifar10_compare.jsonl runs/cifar10_depth.jsonl),
so depth runs are paired with the baseline of the same seed.

Prints three tables:
1. mean ± std per configuration (dataset, model, epochs, base width, method);
2. paired per-seed differences against the reference baseline with the same seed, with
   a 95% confidence interval (Student t). The reference baseline has the same dataset,
   model and base width, the default recipe (lr 0.1, weight decay 5e-4, bf16), and the
   same number of epochs (or ``--ref-epochs N``, to compare runs of another length
   against the N-epoch baseline);
3. accuracy and parameter count at each evaluated width (NetAug runs only),
   plus the per-group random widths that training actually samples.
"""

import argparse
import io
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict

# Two-sided 95% critical values of Student's t for 1..30 degrees of freedom.
T95 = [
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
]


def mean_ci(xs, scale=100.0) -> tuple[float, float]:
    """Mean and half-width of the 95% confidence interval of the mean (nan for one value)."""
    m = statistics.mean(xs) * scale
    if len(xs) < 2:
        return m, float("nan")
    t = T95[len(xs) - 2] if len(xs) - 2 < len(T95) else 2.0
    return m, t * statistics.stdev(xs) * scale / math.sqrt(len(xs))


def ci95(xs, scale=100.0) -> str:
    """Mean and 95% confidence interval of the mean (needs at least two values)."""
    if len(xs) < 2:
        return "n/a"
    m, h = mean_ci(xs, scale)
    return f"{m:+.2f} [{m - h:+.2f}, {m + h:+.2f}]"


def recipe(r: dict) -> str:
    """Suffix for a learning rate, weight decay, precision or warmup other than the defaults (0.1, 5e-4, 16-bit, 0)."""
    out = ""
    if r.get("lr", 0.1) != 0.1:
        out += f" lr{r['lr']:g}"
    if r.get("wd", 5e-4) != 5e-4:
        out += f" wd{r['wd']:g}"
    if r.get("amp", "bf16") not in ("bf16", "fp16"):
        out += f" amp-{r['amp']}"
    if r.get("warmup", 0):
        out += f" warmup{r['warmup']}"
    return out


def width_of(r: dict) -> float:
    """Width of the base model relative to the default TinyResNet."""
    return round(float(r.get("width", 1.0)), 4)


def label_of(r: dict) -> str:
    if r["method"] == "baseline":
        return "baseline" + recipe(r)
    label = "netaug"
    if r.get("depth"):
        kind = "width+depth" if len(r["width_mults"]) > 1 else "depth"
        label = f"netaug {kind} (+{r['depth']}/stage)"
    mults = list(r["width_mults"])
    if mults != [1.0, 2.0, 3.0] and not (r.get("depth") and mults == [1.0]):
        label += " x(" + "/".join(f"{m:g}" for m in mults) + ")"
    label += f"+sort{r['sort_every']}" if r["sort_every"] else ""
    for field, default in (("sort_scope", "all"), ("sort_agg", "sum"), ("sort_opt", "permute")):
        if r.get(field, default) != default:
            label += f" {field.removeprefix('sort_')}-{r[field]}"
    if r.get("alpha", 1.0) != 1.0:
        label += f" alpha{r['alpha']:g}"
    if r.get("stop_aug", 0.0):
        label += f" stop{r['stop_aug']:g}"
    if r.get("init", "base") != "base":
        label += f" init-{r['init']}"
    if r.get("granularity", "group") != "group":
        label += f" {r['granularity']}"
    return label + recipe(r)


def load(paths) -> dict:
    """Group the records of the given jsonl files by (dataset, model, epochs, base width, label)."""
    groups = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = (r["dataset"], r["model"], r["epochs"], width_of(r), label_of(r))
                    if any(x["seed"] == r["seed"] for x in groups[key]):
                        print(f"skipped a second run of seed {r['seed']} for {key} ({path})", file=sys.stderr)
                        continue
                    groups[key].append(r)
    return groups


def paired(groups, key, ref_epochs=None):
    """Pairs (run, reference baseline run) with the same seed, sorted by seed.

    The reference is a baseline with the same dataset, model and base width, and the same
    number of epochs (or ``ref_epochs``). A NetAug run is paired with the baseline of the
    same training recipe (learning rate, weight decay, precision) when one exists, so the
    difference shows NetAug alone; otherwise, and for every baseline, the reference is the
    default-recipe baseline. Empty for the reference itself.
    """
    ds, model, ep, bw, label = key
    if label == "baseline" and ref_epochs and ep < ref_epochs:
        return []
    ref_key = (ds, model, ref_epochs or ep, bw, "baseline")
    if not label.startswith("baseline") and groups.get(key):
        same = (ds, model, ref_epochs or ep, bw, "baseline" + recipe(groups[key][0]))
        if groups.get(same):
            ref_key = same
    if key == ref_key:
        return []
    base = {r["seed"]: r for r in groups.get(ref_key, [])}
    return [(r, base[r["seed"]]) for r in sorted(groups[key], key=lambda r: r["seed"]) if r["seed"] in base]


def is_width(key: str) -> bool:
    try:
        float(key)
        return True
    except ValueError:
        return False


def fmt(xs, scale=100.0):
    m = statistics.mean(xs) * scale
    s = statistics.stdev(xs) * scale if len(xs) > 1 else float("nan")
    return f"{m:.2f} ± {s:.2f}"


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["runs/cifar_results.jsonl"])
    parser.add_argument(
        "--ref-epochs", type=int, default=None, help="pair every configuration with the baseline of this many epochs"
    )
    cli = parser.parse_args()
    groups = load(cli.paths)

    print("## Base-width accuracy\n")
    print(
        "| dataset | model | epochs | base width | method | seeds | params "
        "| test acc (%) | train acc (%) | minutes/run |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (ds, model, ep, bw, label), rs in sorted(groups.items()):
        seeds = ",".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))
        mins = statistics.mean(r["minutes"] for r in rs)
        print(
            f"| {ds} | {model} | {ep} | x{bw:g} | {label} | {seeds} | {rs[0]['params']:,} | "
            f"{fmt([r['test_acc'] for r in rs])} | {fmt([r['train_acc'] for r in rs])} | {mins:.1f} |"
        )

    # Paired differences: same seed means the same initialization of the base weights (except
    # with --init max) and the same data order at the start, so the per-seed difference
    # removes part of the noise.
    print("\n## Paired difference against the reference baseline (percentage points)\n")
    print(
        "Mean [95% confidence interval] of the per-seed difference. The reference is the "
        "baseline of the same dataset, model and base width; for NetAug, the baseline with the "
        "same learning rate, weight decay and precision if there is one, else the default recipe"
        + (f", trained for {cli.ref_epochs} epochs.\n" if cli.ref_epochs else ", with the same number of epochs.\n")
    )
    print(
        "| dataset | model | epochs | base width | method | n | per-seed test diff | test diff: mean ± std | "
        "test diff: mean [95% CI] | train diff: mean [95% CI] |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (ds, model, ep, bw, label), rs in sorted(groups.items()):
        pairs = paired(groups, (ds, model, ep, bw, label), cli.ref_epochs)
        if not pairs:
            continue
        test = [r["test_acc"] - b["test_acc"] for r, b in pairs]
        train = [r["train_acc"] - b["train_acc"] for r, b in pairs]
        per_seed = ", ".join(f"s{r['seed']}: {d * 100:+.2f}" for (r, _), d in zip(pairs, test))
        print(
            f"| {ds} | {model} | {ep} | x{bw:g} | {label} | {len(pairs)} | {per_seed} | {fmt(test)} | "
            f"{ci95(test)} | {ci95(train)} |"
        )

    # Accuracy vs. width: every width exported from NetAug as a standalone plain-PyTorch
    # model, next to the same architecture trained alone at that width.
    for ds, model, ep, bw in sorted({k[:4] for k in groups}):
        netaug_labels = [k[4] for k in sorted(groups) if k[:4] == (ds, model, ep, bw) and k[4].startswith("netaug")]
        widths = sorted(
            {
                w
                for lbl in netaug_labels
                for r in groups[(ds, model, ep, bw, lbl)]
                for w in r.get("width_acc", {})
                if is_width(w)
            },
            key=float,
        )
        if not widths or not netaug_labels:
            continue
        print(f"\n## Exported models at each width ({ds}, {model}, base width x{bw:g}, {ep} epochs)\n")
        print(
            "Test accuracy (%). Widths are relative to the base model. Each NetAug width is "
            "BN-recalibrated, exported with `export(sizes)` (plain PyTorch, no torchnetaug layers) "
            'and evaluated. "trained alone" is the same architecture trained by itself at that '
            "width with the baseline recipe.\n"
        )
        head = (
            ["width", "params", "trained alone"]
            + [f"{lbl} exported" for lbl in netaug_labels]
            + ["max abs(exported - dynamic)"]
        )
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for w in widths:
            alone = groups.get((ds, model, ep, round(bw * float(w), 4), "baseline"), [])
            cells = [f"x{w}"]
            any_run = next(
                r for lbl in netaug_labels for r in groups[(ds, model, ep, bw, lbl)] if w in r.get("width_acc", {})
            )
            cells.append(f"{any_run['width_acc'][w]['params']:,}")
            cells.append(fmt([r["test_acc"] for r in alone]) + f" (n={len(alone)})" if alone else "—")
            diffs = []
            for lbl in netaug_labels:
                rs = [r for r in groups[(ds, model, ep, bw, lbl)] if w in r.get("width_acc", {})]
                if not rs:
                    cells.append("—")
                    continue
                key = "export_test_acc" if all("export_test_acc" in r["width_acc"][w] for r in rs) else "test_acc"
                cells.append(fmt([r["width_acc"][w][key] for r in rs]) + f" (n={len(rs)})")
                diffs += [r["width_acc"][w].get("export_max_abs_diff", float("nan")) for r in rs]
            cells.append(f"{max(diffs):.1e}")
            print("| " + " | ".join(cells) + " |")

    rand = [(k, rs) for k, rs in sorted(groups.items()) if any(r.get("random_acc") for r in rs)]
    if rand:
        print("\n## Per-group random widths (the distribution sampled in training)\n")
        print(
            "| dataset | model | epochs | base width | method | configs | params (min–max) "
            "| test acc (%) mean ± std | min | max |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for (ds, model, ep, bw, label), rs in rand:
            items = [x for r in rs for x in r.get("random_acc", [])]
            accs = [x["test_acc"] for x in items]
            ps = [x["params"] for x in items]
            print(
                f"| {ds} | {model} | {ep} | x{bw:g} | {label} | {len(items)} | {min(ps):,}–{max(ps):,} | "
                f"{fmt(accs)} | {min(accs) * 100:.2f} | {max(accs) * 100:.2f} |"
            )

    deep = [(k, rs) for k, rs in sorted(groups.items()) if any("depth_max" in r.get("width_acc", {}) for r in rs)]
    if deep:
        print("\n## Base width with every extra block on (exported, plain PyTorch)\n")
        print(
            "| dataset | model | epochs | base width | method | params | test acc (%) | max abs(exported - dynamic) |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for (ds, model, ep, bw, label), rs in deep:
            items = [r["width_acc"]["depth_max"] for r in rs if "depth_max" in r.get("width_acc", {})]
            print(
                f"| {ds} | {model} | {ep} | x{bw:g} | {label} | {items[0]['params']:,} | "
                f"{fmt([x['export_test_acc'] for x in items])} (n={len(items)}) | "
                f"{max(x['export_max_abs_diff'] for x in items):.1e} |"
            )

    grads = [(k, rs) for k, rs in sorted(groups.items()) if any(r.get("grad_log") for r in rs)]
    if grads:
        print("\n## Base vs. augmented gradient on the base weights\n")
        print(
            "cos = cosine similarity of dL(base)/dW and dL(aug)/dW on the base entries; "
            "ratio = |dL(aug)/dW| / |dL(base)/dW|. Mean over seeds and over the logged steps "
            "of each epoch range.\n"
        )
        print("| dataset | model | epochs | base width | method | epochs range | cos | ratio |")
        print("|---|---|---|---|---|---|---|---|")
        for (ds, model, ep, bw, label), rs in grads:
            edges = sorted({0, 1, ep // 4, ep // 2, 3 * ep // 4, ep})
            for lo, hi in itertools.pairwise(edges):
                rows = [g for r in rs for g in r.get("grad_log", []) if lo <= g["epoch"] < hi and "cos" in g]
                if rows:
                    cos = statistics.mean(g["cos"] for g in rows)
                    ratio = statistics.mean(g["ratio"] for g in rows)
                    print(f"| {ds} | {model} | {ep} | x{bw:g} | {label} | {lo + 1}–{hi} | {cos:.3f} | {ratio:.2f} |")


if __name__ == "__main__":
    main()
