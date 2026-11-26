"""Report the channel-sorting ablation written by examples/sort_ablation.py.

    uv run python examples/sort_report.py [--lang {zh,en}] [--plot DIR] [results.jsonl ...]

Prints one Markdown table to stdout, one row per (model, setting): test accuracy, the
difference against ``all`` and against ``none`` (paired by seed, percentage points), the
accuracy of each exported wider model, and what the sort log recorded (number of sorts,
groups whose order changed, groups that differ from the default sort, channels moved into
the base model, and the largest change of the max-width output). With more than one seed,
values are means and the table has a column n.

``--plot DIR`` also writes DIR/sort_ablation.png: one panel per model, the test accuracy
difference against ``all`` for each setting.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sort_ablation import SETTINGS
from summarize import is_width, mean_ci

DEFAULTS = {"sort_every": 0, "sort_scope": "all", "sort_agg": "sum", "sort_opt": "permute"}
FLAGS = {"--sort-every": "sort_every", "--sort-scope": "sort_scope", "--sort-agg": "sort_agg", "--sort-opt": "sort_opt"}
REF, NONE = "all", "none"

HEADERS = {
    "zh": {
        "model": "模型",
        "setting": "設定",
        "n": "n",
        "test": "test acc (%)",
        "vs_ref": "對 all（百分點）",
        "vs_none": "對 none（百分點）",
        "export": "匯出 ×{w}",
        "sorts": "排序次數",
        "changed": "每次排序改變順序的群組",
        "diff": "每次排序與預設排序不同的群組",
        "moved": "移入 base 的通道（全部排序合計）",
        "out": "最大寬度輸出的最大變化",
        "minutes": "分鐘",
        "caption": "{ds}，{ep} epoch，seed {seeds}。",
        "mean": "數值是 {n} 個 seed 的平均；差異是同 seed 配對相減後的平均。",
    },
    "en": {
        "model": "model",
        "setting": "setting",
        "n": "n",
        "test": "test acc (%)",
        "vs_ref": "vs all (points)",
        "vs_none": "vs none (points)",
        "export": "exported ×{w}",
        "sorts": "sorts",
        "changed": "groups reordered per sort",
        "diff": "groups different from the default sort, per sort",
        "moved": "channels moved into base (all sorts)",
        "out": "max change of the max-width output",
        "minutes": "minutes",
        "caption": "{ds}, {ep} epochs, seed {seeds}.",
        "mean": "Values are means over {n} seeds; differences are means of the per-seed paired differences.",
    },
}
DATASETS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}


def flags_of(args: list[str]) -> tuple:
    """(sort_every > 0, scope, agg, opt) for a list of cifar_netaug.py sort flags."""
    values = dict(DEFAULTS)
    for flag, value in zip(args[::2], args[1::2]):
        values[FLAGS[flag]] = int(value) if flag == "--sort-every" else value
    if not values["sort_every"]:
        return (False, "all", "sum", "permute")
    return (True, values["sort_scope"], values["sort_agg"], values["sort_opt"])


NAMES = {flags_of(args): name for name, args in SETTINGS.items()}
ORDER = list(SETTINGS)


def setting_of(r: dict) -> str | None:
    """The sort_ablation.py setting name of a record, or None for a record outside the ablation."""
    if r.get("method") != "netaug" or r.get("sort_every", 0) not in (0, 1):
        return None
    if not r.get("sort_every"):
        return NONE
    key = (True, r.get("sort_scope", "all"), r.get("sort_agg", "sum"), r.get("sort_opt", "permute"))
    return NAMES.get(key)


def load(paths) -> dict:
    """Records grouped by (dataset, model, epochs) and then setting; one record per seed."""
    groups: dict = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                setting = setting_of(r)
                if setting is None:
                    print(f"skipped a record outside the ablation ({path})", file=sys.stderr)
                    continue
                rows = groups[(r["dataset"], r["model"], r["epochs"])][setting]
                if any(x["seed"] == r["seed"] for x in rows):
                    print(
                        f"skipped a second run of seed {r['seed']} for {r['model']} {setting} ({path})", file=sys.stderr
                    )
                    continue
                rows.append(r)
    return groups


def paired_diff(rs: list[dict], refs: list[dict]) -> tuple[float, float] | None:
    """Mean and 95% half-width of the per-seed test accuracy difference (points), or None."""
    ref = {r["seed"]: r["test_acc"] for r in refs}
    diffs = [r["test_acc"] - ref[r["seed"]] for r in rs if r["seed"] in ref]
    return mean_ci(diffs) if diffs else None


def signed(x: float) -> str:
    return f"{x:+.2f}".replace("-", "−")


def short(x: float) -> str:
    """One decimal, without a trailing .0."""
    return f"{x:.1f}".removesuffix(".0")


def per_sort(rs: list[dict], field: str) -> float | None:
    values = [e[field] for r in rs for e in r.get("sort_log", []) if field in e]
    return statistics.mean(values) if values else None


def table(groups: dict, lang: str) -> str:
    h = HEADERS[lang]
    facets = sorted(groups)
    seeds = sorted({r["seed"] for f in facets for rs in groups[f].values() for r in rs})
    many = any(len(rs) > 1 for f in facets for rs in groups[f].values())
    widths = sorted(
        {
            w
            for f in facets
            for rs in groups[f].values()
            for r in rs
            for w in r.get("width_acc", {})
            if is_width(w) and float(w) > 1
        },
        key=float,
    )
    one_facet = len({(ds, ep) for ds, _, ep in facets}) == 1

    head = [h["model"], h["setting"]] + ([h["n"]] if many else []) + [h["test"], h["vs_ref"], h["vs_none"]]
    head += [h["export"].format(w=f"{float(w):g}") for w in widths]
    head += [h["sorts"], h["changed"], h["diff"], h["moved"], h["out"], h["minutes"]]
    lines = []
    if one_facet and facets:
        ds, _, ep = facets[0]
        lines.append(h["caption"].format(ds=DATASETS.get(ds, ds), ep=ep, seeds=", ".join(map(str, seeds))))
        if many:
            lines.append(h["mean"].format(n=len(seeds)))
        lines.append("")
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for facet in facets:
        ds, model, ep = facet
        name = model if one_facet else f"{model} ({DATASETS.get(ds, ds)}, {ep} ep)"
        by_setting = groups[facet]
        for setting in [s for s in ORDER if s in by_setting]:
            rs = sorted(by_setting[setting], key=lambda r: r["seed"])
            cells = [name, f"`{setting}`"] + ([str(len(rs))] if many else [])
            cells.append(f"{statistics.mean(r['test_acc'] for r in rs) * 100:.2f}")
            for ref in (REF, NONE):
                d = None if setting == ref else paired_diff(rs, by_setting.get(ref, []))
                cells.append("—" if d is None else signed(d[0]))
            for w in widths:
                vals = [
                    r["width_acc"][w].get("export_test_acc", r["width_acc"][w]["test_acc"])
                    for r in rs
                    if w in r.get("width_acc", {})
                ]
                cells.append(f"{statistics.mean(vals) * 100:.2f}" if vals else "—")
            sorts = [len(r.get("sort_log", [])) for r in rs]
            cells.append(f"{statistics.mean(sorts):g}")
            changed, groups_in, diff = (per_sort(rs, k) for k in ("sorted_groups", "groups", "diff_from_default"))
            cells.append("—" if changed is None or groups_in is None else f"{short(changed)} / {short(groups_in)}")
            cells.append("—" if diff is None else short(diff))
            logged = [r for r in rs if r.get("sort_log")]
            moved = [
                sum(e["moved_into_base"] for e in r["sort_log"])
                for r in logged
                if all("moved_into_base" in e for e in r["sort_log"])
            ]
            cells.append(f"{statistics.mean(moved):g}" if moved else "—")
            outs = [e["max_abs_diff"] for r in logged for e in r["sort_log"] if "max_abs_diff" in e]
            cells.append(f"{max(outs):.1e}" if outs else "—")
            cells.append(f"{statistics.mean(r['minutes'] for r in rs):.1f}")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(groups: dict, out: Path) -> Path | None:
    """One panel per model: test accuracy difference against ``all`` for each setting."""
    from plot_results import ALONE, INK2, MUTED, NETAUG, THRESHOLD, panels, save

    facets = [f for f in sorted(groups) if REF in groups[f]]
    if not facets:
        print(f"no figure: no model has the reference setting {REF!r}", file=sys.stderr)
        return None
    rows = max(len(groups[f]) for f in facets)
    fig, axes = panels(len(facets), w=4.6, h=max(2.4, 0.36 * rows + 1.2))
    one_facet = len({(ds, ep) for ds, _, ep in facets}) == 1
    for ax, facet in zip(axes, facets):
        ds, model, ep = facet
        by_setting = groups[facet]
        settings = [s for s in ORDER if s in by_setting]
        ys = list(range(len(settings)))[::-1]
        extent = [THRESHOLD * 2]
        for y, setting in zip(ys, settings):
            if setting == REF:
                ax.annotate(
                    "reference", (0, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK2
                )
                continue
            d = paired_diff(by_setting[setting], by_setting[REF])
            if d is None:
                continue
            m, hw = d
            hw = 0.0 if math.isnan(hw) else hw
            extent.append(abs(m) + hw)
            color = ALONE if setting == NONE else NETAUG
            ax.barh(y, m, height=0.6, color=color, xerr=hw or None, ecolor=INK2)
            ax.annotate(
                signed(m),
                (m + hw if m >= 0 else m - hw, y),  # past the end of the error bar
                xytext=(4 if m >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if m >= 0 else "right",
                va="center",
                fontsize=8,
                color=INK2,
            )
        ax.axvline(0, color=MUTED, linewidth=1)
        for x in (-THRESHOLD, THRESHOLD):
            ax.axvline(x, color=MUTED, linewidth=1, linestyle="--")
        span = max(extent) * 1.4  # room for the value labels
        ax.set_xlim(-span, span)
        ax.set_yticks(ys, settings)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("test accuracy difference against all (points)")
        n = len({r["seed"] for rs in by_setting.values() for r in rs})
        where = "" if one_facet else f", {DATASETS.get(ds, ds)}, {ep} epochs"
        ax.set_title(f"{model}{where} (n={n})")
    ds, _, ep = facets[0]
    head = f"{DATASETS.get(ds, ds)}, {ep} epochs: " if one_facet else ""
    many = any(len(rs) > 1 for f in facets for rs in groups[f].values())
    stat = "paired by seed, mean and 95% CI; " if many else ""
    fig.suptitle(f"{head}channel-sorting ablation ({stat}dashed lines at ±{THRESHOLD:g} points)", y=1.02)
    fig.subplots_adjust(wspace=0.45)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, "sort_ablation.png")
    return out / "sort_ablation.png"


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", default=["runs/sort_ablation.jsonl"])
    p.add_argument("--lang", default="zh", choices=list(HEADERS), help="language of the header row")
    p.add_argument("--plot", metavar="DIR", default=None, help="also write DIR/sort_ablation.png")
    args = p.parse_args()
    groups = load(args.paths)
    if not groups:
        sys.exit("no ablation records in " + ", ".join(args.paths))
    print(table(groups, args.lang))
    if args.plot:
        stdout, sys.stdout = sys.stdout, sys.stderr  # keep the "wrote ..." line out of the table
        try:
            plot(groups, Path(args.plot))
        finally:
            sys.stdout = stdout


if __name__ == "__main__":
    main()
