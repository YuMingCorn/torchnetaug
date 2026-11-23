"""Plot CIFAR results written by cifar_netaug.py (the same grouping as summarize.py).

    uv run python examples/plot_results.py [--out runs/plots] [--variant LABEL] [--ref-epochs N] results.jsonl ...

Writes PNG files to --out:
- acc_vs_params.png      test accuracy against parameter count: plain models trained alone
                         at several widths, NetAug base models, and the wider models
                         exported from NetAug (one panel per dataset and epoch count;
                         the NetAug configuration is --variant, default "netaug");
- acc_vs_width_mult.png  test accuracy against width multiplier, for each NetAug base width:
                         the exported NetAug model next to the same architecture trained alone;
- paired_diff.png        per-seed difference against the reference baseline (test and train
                         accuracy), mean and 95% confidence interval; the dashed line at +0.5
                         points is the "material effect" threshold;
- gradient.png           cosine and norm ratio of the augmented-network gradient against the
                         base gradient on the base weights, per epoch (runs with --grad-log-every).
A figure is skipped when the files hold no data for it.
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import is_width, load, mean_ci, paired  # noqa: E402

# Colors (validated for color-vision deficiency as a set of three): one per role.
ALONE, NETAUG = "#2a78d6", "#eb6834"
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]  # light to dark blue, for ordered series
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
THRESHOLD = 0.5  # percentage points: the smallest test gain we call material

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 6, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False, "errorbar.capsize": 0,
})


def acc(rs, key="test_acc"):
    return mean_ci([r[key] for r in rs])


def exported(rs, w):
    """(params, mean, ci) of the model exported at width multiplier w, or None."""
    rs = [r for r in rs if w in r.get("width_acc", {})]
    if not rs:
        return None
    key = "export_test_acc" if all("export_test_acc" in r["width_acc"][w] for r in rs) else "test_acc"
    m, h = mean_ci([r["width_acc"][w][key] for r in rs])
    return rs[0]["width_acc"][w]["params"], m, h, len(rs)


def bar(ax, x, m, h, color, **kw):
    """A point with a 95% confidence bar (no bar for a single seed)."""
    ax.errorbar(x, m, yerr=None if h != h else h, color=color, fmt="o", elinewidth=1.5, **kw)


def label_end(ax, x, y, text, color=INK2):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8, color=color)


def log_ticks(ax, values):
    """Ticks at 1, 2 and 5 times powers of ten inside the data range, labelled 20k, 1M, ..."""
    lo, hi = min(values) / 1.15, max(values) * 1.15
    ticks = [m * 10 ** k for k in range(2, 9) for m in (1, 2, 5) if lo <= m * 10 ** k <= hi]
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks, [f"{t / 1e6:g}M" if t >= 1e6 else f"{t / 1e3:g}k" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig, out, name):
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def panels(n, w=4.2, h=3.4):
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def plot_acc_vs_params(groups, out, variant):
    facets = sorted({(ds, model, ep) for (ds, model, ep, bw, label) in groups if label == variant}
                    | {(ds, model, ep) for (ds, model, ep, bw, label) in groups
                       if label == "baseline" and sum(k[:3] == (ds, model, ep) and k[4] == "baseline"
                                                      for k in groups) > 1})
    if not facets:
        return
    fig, axes = panels(len(facets))
    for ax, (ds, model, ep) in zip(axes, facets):
        alone = sorted((rs[0]["params"], *acc(rs)) for (d, mo, e, bw, l), rs in groups.items()
                       if (d, mo, e, l) == (ds, model, ep, "baseline"))
        if alone:
            ax.plot([p for p, _, _ in alone], [m for _, m, _ in alone], color=ALONE, label="trained alone")
            for p, m, h in alone:
                bar(ax, p, m, h, ALONE)
        bases = sorted(bw for (d, mo, e, bw, l) in groups if (d, mo, e, l) == (ds, model, ep, variant))
        base_pts = []
        for i, bw in enumerate(bases):
            rs = groups[(ds, model, ep, bw, variant)]
            m, h = acc(rs)
            base_pts.append((rs[0]["params"], m))
            widths = sorted({w for r in rs for w in r.get("width_acc", {}) if is_width(w) and float(w) > 1},
                            key=float)
            pts = [exported(rs, w) for w in widths]
            if pts:
                ax.plot([rs[0]["params"]] + [p[0] for p in pts], [m] + [p[1] for p in pts], color=NETAUG,
                        linestyle="--", linewidth=1.5, label="wider models exported from NetAug" if i == 0 else None)
                for p, pm, ph, _ in pts:
                    bar(ax, p, pm, ph, NETAUG, mfc=SURFACE, mew=1.5)
            bar(ax, rs[0]["params"], m, h, NETAUG, zorder=3)
        if base_pts:
            ax.plot(*zip(*base_pts), color=NETAUG, label=f"{variant} base model", zorder=3)
        ax.set_xscale("log")
        log_ticks(ax, [x for line in ax.get_lines() for x in line.get_xdata()])
        ax.set_xlabel("parameters (log scale)")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(f"{ds}, {model}, {ep} epochs")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Test accuracy against model size (mean, 95% confidence interval over seeds)", y=1.01)
    save(fig, out, "acc_vs_params.png")


def plot_acc_vs_width_mult(groups, out, variant):
    facets = sorted(k for k, rs in groups.items()
                    if k[4] == variant and any(is_width(w) for r in rs for w in r.get("width_acc", {})))
    if not facets:
        return
    fig, axes = panels(len(facets), w=3.6)
    for ax, (ds, model, ep, bw, label) in zip(axes, facets):
        rs = groups[(ds, model, ep, bw, label)]
        widths = sorted({w for r in rs for w in r.get("width_acc", {}) if is_width(w)}, key=float)
        net = [(float(w), exported(rs, w)) for w in widths]
        net = [(x, p) for x, p in net if p]
        ax.plot([x for x, _ in net], [p[1] for _, p in net], color=NETAUG)
        for x, (_, m, h, _) in net:
            bar(ax, x, m, h, NETAUG)
        label_end(ax, net[-1][0], net[-1][1][1], f"{variant} (n={net[-1][1][3]})")
        alone = [(float(w), groups.get((ds, model, ep, round(bw * float(w), 4), "baseline"))) for w in widths]
        alone = [(x, acc(a), len(a)) for x, a in alone if a]
        if alone:
            ax.plot([x for x, _, _ in alone], [m for _, (m, _), _ in alone], color=ALONE)
            for x, (m, h), _ in alone:
                bar(ax, x, m, h, ALONE)
            label_end(ax, alone[-1][0], alone[-1][1][0], f"trained alone (n={alone[-1][2]})")
        ax.set_xticks([x for x, _ in net], [f"x{x:g}" for x, _ in net])
        ax.set_xlim(min(x for x, _ in net) - 0.2, max(x for x, _ in net) + 1.0)
        ax.set_xlabel(f"width multiplier (x1 = base width {bw:g})")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(f"{ds}, base width {bw:g}, {ep} epochs")
    fig.suptitle("Wider models exported from NetAug against the same models trained alone", y=1.01)
    save(fig, out, "acc_vs_width_mult.png")


def verdict(m, h):
    if h != h:
        return "one seed"
    if m - h > 0:
        return "gain"
    if m + h < THRESHOLD:
        return "no material gain"
    return "inconclusive"


def plot_paired(groups, out, ref_epochs):
    rows = []
    for key in sorted(groups):
        pairs = paired(groups, key, ref_epochs)
        if pairs:
            test = mean_ci([r["test_acc"] - b["test_acc"] for r, b in pairs])
            train = mean_ci([r["train_acc"] - b["train_acc"] for r, b in pairs])
            ds, model, ep, bw, label = key
            rows.append((f"{ds} x{bw:g} {ep}ep  {label}  (n={len(pairs)})", test, train))
    if not rows:
        return
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(10, max(2.6, 0.42 * len(rows) + 1.4)), sharey=True,
                                     gridspec_kw={"width_ratios": [1.3, 1]})
    ys = list(range(len(rows)))[::-1]
    for y, (name, (tm, th), (rm, rh)) in zip(ys, rows):
        ax_t.errorbar(tm, y, xerr=None if th != th else th, fmt="o", color=NETAUG, elinewidth=1.5)
        ax_r.errorbar(rm, y, xerr=None if rh != rh else rh, fmt="o", color=NETAUG, elinewidth=1.5)
        ax_t.annotate(verdict(tm, th), (1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                      textcoords="offset points", va="center", fontsize=8, color=INK2)
    for ax in (ax_t, ax_r):
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.grid(axis="y", visible=False)
    ax_t.axvline(THRESHOLD, color=MUTED, linewidth=1, linestyle="--")
    ax_t.annotate(f"+{THRESHOLD:g}: material gain", (THRESHOLD, 1.0), xycoords=("data", "axes fraction"),
                  xytext=(3, 2), textcoords="offset points", fontsize=8, color=INK2)
    ax_t.set_yticks(ys, [r[0] for r in rows])
    ax_t.set_xlabel("test accuracy difference (points)")
    ax_r.set_xlabel("train accuracy difference (points)")
    ref = f"{ref_epochs}-epoch baseline" if ref_epochs else "baseline with the same epochs"
    fig.suptitle(f"Difference against the {ref}, paired by seed (mean, 95% CI)", y=1.02)
    fig.subplots_adjust(wspace=0.45)
    save(fig, out, "paired_diff.png")


def plot_gradient(groups, out):
    series = [(k, rs) for k, rs in sorted(groups.items())
              if any("cos" in g for r in rs for g in r.get("grad_log", []))]
    if not series:
        return
    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(10, 3.6))
    colors = {1: [ORDINAL[1]], 2: [ORDINAL[0], ORDINAL[2]], 3: ORDINAL}.get(len(series), [None] * len(series))
    for color, ((ds, model, ep, bw, label), rs) in zip(colors, series):
        by_epoch = defaultdict(list)
        for r in rs:
            for g in r.get("grad_log", []):
                if "cos" in g:
                    by_epoch[g["epoch"]].append(g)
        epochs = sorted(by_epoch)
        name = f"{ds} x{bw:g} {label}"
        for ax, field in ((ax_c, "cos"), (ax_r, "ratio")):
            ys = [statistics.mean(g[field] for g in by_epoch[e]) for e in epochs]
            ax.plot([e + 1 for e in epochs], ys, color=color, label=name)
            label_end(ax, epochs[-1] + 1, ys[-1], f"x{bw:g}")
    ax_c.set_ylabel("cosine (aug gradient, base gradient)")
    ax_r.set_ylabel("|aug gradient| / |base gradient|")
    for ax in (ax_c, ax_r):
        ax.set_xlabel("epoch")
    ax_c.axhline(0, color=MUTED, linewidth=1)
    ax_r.axhline(1, color=MUTED, linewidth=1)
    for ax in (ax_c, ax_r):
        ax.set_xlim(right=ax.get_xlim()[1] + 4)
    ax_c.legend(fontsize=8, loc="lower left")
    fig.suptitle("Gradient of the augmented network on the base weights (mean over seeds and logged steps)",
                 y=1.02)
    save(fig, out, "gradient.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default="runs/plots")
    p.add_argument("--variant", default="netaug",
                   help="NetAug configuration (a label from summarize.py, e.g. 'netaug init-max') "
                        "drawn in the accuracy figures; paired_diff.png shows every configuration")
    p.add_argument("--ref-epochs", type=int, default=None,
                   help="pair every configuration with the baseline of this many epochs")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = load(args.paths)
    plot_acc_vs_params(groups, out, args.variant)
    plot_acc_vs_width_mult(groups, out, args.variant)
    plot_paired(groups, out, args.ref_epochs)
    plot_gradient(groups, out)


if __name__ == "__main__":
    main()
