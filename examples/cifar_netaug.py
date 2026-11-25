"""Compare plain training with NetAug training of a tiny model on CIFAR.

Example:
    uv run python examples/cifar_netaug.py --method baseline --seed 0
    uv run python examples/cifar_netaug.py --method netaug --seed 0
    uv run python examples/cifar_netaug.py --method netaug --seed 0 --sort-every 10
    uv run python examples/cifar_netaug.py --method netaug --seed 0 --depth 1 --width-mults 1.0
    uv run python examples/cifar_netaug.py --method netaug --seed 0 --dataset cifar100 --width 0.5

Each run appends one JSON line to ``--out`` (default ``runs/cifar_results.jsonl``).
The whole dataset lives on the GPU and augmentation (random crop + flip) runs there too,
so no DataLoader workers are needed.

Training runs the forward pass in bfloat16 by default (``--amp fp16`` for GPUs without
bfloat16, such as the T4, with a gradient scaler; ``--amp off`` for float32). The loss,
BN calibration and all evaluation run in float32.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from torchnetaug import DynamicConv2d, DynamicLinear, NetAugModel, netaug_step, recalibrate_bn

MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}


def load(name: str, root: str, device: torch.device):
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    out = []
    for train in (True, False):
        ds = cls(root, train=train, download=True)
        x = torch.tensor(ds.data, device=device).permute(0, 3, 1, 2).float().div_(255)
        mean = torch.tensor(MEAN[name], device=device).view(1, 3, 1, 1)
        std = torch.tensor(STD[name], device=device).view(1, 3, 1, 1)
        out.append(((x - mean) / std, torch.tensor(ds.targets, device=device)))
    return out


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random crop 32 with padding 4, and random horizontal flip, on the GPU."""
    n = x.shape[0]
    padded = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (n,), device=x.device, generator=gen)
    j = torch.randint(0, 9, (n,), device=x.device, generator=gen)
    rows = (i.view(n, 1) + torch.arange(32, device=x.device)).view(n, 1, 32, 1).expand(n, 3, 32, 32)
    cols = (j.view(n, 1) + torch.arange(32, device=x.device)).view(n, 1, 1, 32).expand(n, 3, 32, 32)
    out = padded[
        torch.arange(n, device=x.device).view(n, 1, 1, 1), torch.arange(3, device=x.device).view(1, 3, 1, 1), rows, cols
    ]
    flip = torch.rand(n, device=x.device, generator=gen) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def build_model(name: str, num_classes: int, width: float = 1.0) -> nn.Module:
    m = torchvision.models
    if name == "mbv2_035":
        return m.mobilenet_v2(width_mult=0.35, num_classes=num_classes)
    if name == "mbv2_050":
        return m.mobilenet_v2(width_mult=0.5, num_classes=num_classes)
    if name == "resnet_tiny":
        return TinyResNet(tuple(max(1, round(w * width)) for w in (8, 16, 32, 64)), num_classes)
    raise ValueError(name)


class TinyResNet(nn.Module):
    """CIFAR-style ResNet-18: 3x3 stem, no max-pool, two BasicBlocks per stage.

    ``widths=(8, 16, 32, 64)`` is ResNet-18 at 1/8 width (176,402 parameters for 10 classes).
    """

    def __init__(self, widths=(8, 16, 32, 64), num_classes: int = 10) -> None:
        super().__init__()
        from torchvision.models.resnet import BasicBlock

        w = list(widths)
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, padding=1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        layers, c_in = [], w[0]
        for i, c in enumerate(w):
            stride = 1 if i == 0 else 2
            down = None
            if stride != 1 or c_in != c:
                down = nn.Sequential(nn.Conv2d(c_in, c, 1, stride, bias=False), nn.BatchNorm2d(c))
            layers += [BasicBlock(c_in, c, stride, down), BasicBlock(c, c)]
            c_in = c
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w[-1], num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.layers(self.stem(x))), 1))


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, bs: int = 1000) -> float:
    """Top-1 accuracy in float32 (no autocast)."""
    model.eval()
    correct = 0
    for k in range(0, x.shape[0], bs):
        correct += (model(x[k : k + bs]).argmax(1) == y[k : k + bs]).sum().item()
    return correct / x.shape[0]


def run_key(r: dict) -> tuple:
    return (
        r["dataset"],
        r["model"],
        r["epochs"],
        r["method"],
        r["sort_every"],
        r["seed"],
        tuple(r["width_mults"]),
        r["alpha"],
        r["stop_aug"],
        r.get("width", 1.0),
        r.get("depth", 0),
        r.get("lr", 0.1),
        r.get("wd", 5e-4),
        r.get("amp", "bf16"),
        r.get("init", "base"),
        r.get("granularity", "group"),
        r.get("warmup", 0),
        r.get("sort_scope", "all"),
        r.get("sort_agg", "sum"),
        r.get("sort_opt", "permute"),
    )


def base_masks(model: NetAugModel) -> dict:
    """For each resized parameter, a bool mask of the entries the base model uses.

    Built by differentiating the base slice of each dynamic layer, so it follows the
    same indexing as the forward pass (including concat segments). The extra blocks of
    depth augmentation are not part of the base model: all their entries are masked out.
    """
    masks = {}
    with model.mode_ctx("base"), torch.enable_grad():
        for _, layer in model.dynamic_layers():
            for attr in ("weight", "bias"):
                p = getattr(layer, attr, None)
                if p is None:
                    continue
                if isinstance(layer, (DynamicConv2d, DynamicLinear)):
                    sliced = layer.active_weight() if attr == "weight" else layer.active_bias()
                else:  # batch norm: input channels only
                    sliced = layer._select_in(p, 0)
                assert sliced is not None
                (g,) = torch.autograd.grad(sliced.sum(), p)
                masks[id(p)] = g != 0
    for path in model._depth_paths:
        for p in model._stack(path).extra.parameters():
            masks[id(p)] = torch.zeros_like(p, dtype=torch.bool)
    return masks


class GradProbe:
    """Split one NetAug step's gradient into the base and the augmented part.

    ``netaug_step`` accumulates ``dL(base)/dW`` and then ``alpha * dL(aug)/dW`` into the
    same ``.grad``. Snapshots after each backward give both parts. Only the entries of the
    base model (``W_t`` in the paper) are compared: cosine similarity and norm ratio
    ``|g_aug| / |g_base|``.
    """

    def __init__(self, model: NetAugModel, alpha: float, backward=lambda loss: loss.backward()) -> None:
        self._backward = backward
        self.params = [p for p in model.parameters() if p.requires_grad]
        masks = base_masks(model)
        self.masks = [masks.get(id(p)) for p in self.params]
        self.alpha = alpha
        self.snaps: list = []
        self.cos: list = []
        self.ratio: list = []

    def _flat_base(self) -> torch.Tensor:
        parts = []
        for p, m in zip(self.params, self.masks):
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            parts.append((g[m] if m is not None else g.reshape(-1)).float())
        return torch.cat(parts)

    def backward(self, loss: torch.Tensor) -> None:
        self._backward(loss)  # a gradient scaler scales both parts alike: cos and ratio stay
        self.snaps.append(self._flat_base())

    def finish_step(self) -> None:
        if len(self.snaps) == 2:
            g_base = self.snaps[0]
            g_aug = (self.snaps[1] - self.snaps[0]) / self.alpha
            cos = F.cosine_similarity(g_base, g_aug, dim=0).item()
            ratio = (g_aug.norm() / g_base.norm()).item()
            if math.isfinite(cos) and math.isfinite(ratio):  # an fp16 overflow step is skipped
                self.cos.append(cos)
                self.ratio.append(ratio)
        self.snaps = []

    def epoch_summary(self) -> dict:
        out = (
            {"cos": sum(self.cos) / len(self.cos), "ratio": sum(self.ratio) / len(self.ratio), "n": len(self.cos)}
            if self.cos
            else {}
        )
        self.cos, self.ratio = [], []
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["baseline", "netaug"], required=True)
    p.add_argument("--model", default="resnet_tiny", choices=["mbv2_035", "mbv2_050", "resnet_tiny"])
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument(
        "--warmup", type=int, default=0, help="epochs of linear learning-rate warmup before the cosine decay"
    )
    p.add_argument(
        "--amp",
        default="bf16",
        choices=["bf16", "fp16", "off"],
        help="mixed precision of the training forward pass (fp16 uses a gradient scaler)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width-mults", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--stop-aug", type=float, default=0.0, help="fraction of final epochs trained without augmentation")
    p.add_argument("--sort-every", type=int, default=0, help="sort channels every N epochs (0 = never)")
    p.add_argument(
        "--sort-scope",
        default="all",
        choices=["all", "closed", "inner"],
        help="NetAug with --sort-every > 0 only: the groups to sort (every free group; inner and "
        "closed groups; inner groups only; see NetAugModel.group_kinds)",
    )
    p.add_argument(
        "--sort-agg",
        default="sum",
        choices=["sum", "mean", "single"],
        help="NetAug with --sort-every > 0 only: how the readers' L1 importances are combined. "
        "sum: the library default; mean: each reader's importance divided by its weight "
        "elements per channel, then summed; single: only the first reader",
    )
    p.add_argument(
        "--sort-opt",
        default="permute",
        choices=["permute", "keep", "rebuild"],
        help="NetAug with --sort-every > 0 only: the optimizer state at a sort. permute: permuted "
        "with the weights; keep: not changed (as the official code); rebuild: cleared",
    )
    p.add_argument(
        "--eval-widths",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0],
        help="NetAug only: also report accuracy at these global width multipliers",
    )
    p.add_argument(
        "--eval-random",
        type=int,
        default=3,
        help="NetAug only: also report accuracy of this many per-group random widths",
    )
    p.add_argument(
        "--width",
        type=float,
        default=1.0,
        help="TinyResNet only: width of the base model relative to (8, 16, 32, 64). For the "
        "baseline this is the model trained alone; for NetAug it is the base model",
    )
    p.add_argument(
        "--init",
        default="base",
        choices=["base", "max"],
        help="NetAug only. base: the base weights keep the initialization of the plain model "
        "(same as the baseline of the same seed) and the added channels get the default "
        "initialization. max: the default initialization is applied again to the whole "
        "max-width network, as the official code does; the base weights then differ "
        "from the baseline of the same seed",
    )
    p.add_argument(
        "--granularity",
        default="group",
        choices=["group", "global"],
        help="NetAug only: sample a width for each channel group, or one multiplier for the whole model",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=0,
        help="depth augmentation: extra blocks after each TinyResNet stage (netaug only)",
    )
    p.add_argument(
        "--grad-log-every",
        type=int,
        default=0,
        help="NetAug only: every N steps, compare the base and augmented gradients "
        "on the base weights (cosine, norm ratio); 0 = never",
    )
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/cifar_results.jsonl")
    p.add_argument("--skip-existing", action="store_true", help="skip if --out already has this run")
    args = p.parse_args()
    sorting = args.method == "netaug" and args.sort_every > 0
    if not sorting:  # the sort flags have no effect: store the defaults
        args.sort_scope, args.sort_agg, args.sort_opt = "all", "sum", "permute"

    key = run_key(
        {
            "dataset": args.dataset,
            "model": args.model,
            "epochs": args.epochs,
            "method": args.method,
            "sort_every": args.sort_every if args.method == "netaug" else 0,
            "seed": args.seed,
            "width_mults": args.width_mults,
            "alpha": args.alpha,
            "stop_aug": args.stop_aug,
            "width": args.width,
            "depth": args.depth,
            "lr": args.lr,
            "wd": args.wd,
            "amp": args.amp,
            "init": args.init,
            "granularity": args.granularity,
            "warmup": args.warmup,
            "sort_scope": args.sort_scope,
            "sort_agg": args.sort_agg,
            "sort_opt": args.sort_opt,
        }
    )
    if args.depth and (args.method != "netaug" or args.model != "resnet_tiny"):
        p.error("--depth is only for --method netaug --model resnet_tiny")
    if args.width != 1.0 and args.model != "resnet_tiny":
        p.error("--width is only for --model resnet_tiny")
    if args.method != "netaug" and (args.init != "base" or args.granularity != "group"):
        p.error("--init and --granularity are only for --method netaug")
    if args.skip_existing and Path(args.out).exists():
        with open(args.out, encoding="utf-8") as f:
            if any(run_key(json.loads(line)) == key for line in f if line.strip()):
                print("already done:", key)
                return

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = random.Random(args.seed)
    (xtr, ytr), (xte, yte) = load(args.dataset, args.data, device)
    num_classes = int(ytr.max()) + 1

    base = build_model(args.model, num_classes, args.width).to(device)
    n_base = sum(p.numel() for p in base.parameters())
    if args.method == "netaug":
        # depth: the second block of each stage (layers.1, .3, .5, .7) gets extra copies
        depth = {f"layers.{2 * s + 1}": args.depth for s in range(4)} if args.depth else None
        netaug_model = NetAugModel(base, xtr[:2], width_mults=args.width_mults, depth=depth)
        if args.init == "max":
            # Initialize the largest network as a whole (the official code does this). The
            # default initializer scales with the fan-in, which is now the max-width fan-in
            # for the base channels too. BN weights stay 1 / 0 (0 in residual extra blocks).
            for _, layer in netaug_model.dynamic_layers():
                if isinstance(layer, (DynamicConv2d, DynamicLinear)):
                    layer.reset_parameters()
        print(netaug_model.graph.summary())
        model: nn.Module = netaug_model
    else:
        model = base

    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
    )
    steps_per_epoch = xtr.shape[0] // args.bs
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch * args.warmup

    def lr_factor(s):  # linear warmup over `warm` steps, then cosine decay to 0
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    aug_until = round(args.epochs * (1 - args.stop_aug))

    def crit(out, y):  # float32 loss even though the forward pass runs in bf16
        return F.cross_entropy(out.float(), y)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")

    def fwd(m, inp):
        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            return m(inp)

    def backward(loss):
        scaler.scale(loss).backward()

    sort_log = []

    def is_identity(perm: torch.Tensor) -> bool:
        return torch.equal(perm, torch.arange(perm.numel(), device=perm.device))

    def sort_groups(m: NetAugModel) -> set[int]:
        if args.sort_scope == "all":
            return set(m.graph.free_groups())
        kinds = {"inner"} if args.sort_scope == "inner" else {"inner", "closed"}
        return {g for g, info in m.group_kinds().items() if info.kind in kinds}

    def sort_channels(m: NetAugModel) -> dict[int, torch.Tensor]:
        """Sort with --sort-scope, --sort-agg and --sort-opt; returns the permutations."""
        o = opt if args.sort_opt == "permute" else None
        if args.sort_agg == "sum":
            perms = m.sort_channels(optimizer=o, scope=args.sort_scope)
        else:
            groups = sort_groups(m)
            perms = {}
            for g, per in m.channel_importance(per_reader=True).items():
                if g not in groups:
                    continue
                parts = []
                for name, imp in per.items():
                    w = cast(nn.Conv2d | nn.Linear, m.model.get_submodule(name)).weight
                    dim = 1 if g in m.graph.layers[name].in_groups else 0  # 0: producer fallback
                    parts.append(imp / (w.numel() // w.shape[dim]) if args.sort_agg == "mean" else imp)
                if args.sort_agg == "single":
                    parts = parts[:1]
                imp = sum(parts[1:], parts[0]) if parts else torch.zeros(m.max_sizes[g], device=device)
                perms[g] = imp.argsort(descending=True, stable=True)
            m.apply_permutations(perms, o)
        if args.sort_opt == "rebuild":
            opt.state.clear()  # momentum starts again; lr schedule and param_groups unchanged
        return perms

    @torch.no_grad()
    def sort_and_log(epoch: int) -> None:
        """Sort channels and record evidence that the sort (a) moved channels across the
        base boundary and (b) kept the max-width function on the real trained weights."""
        assert isinstance(model, NetAugModel)
        probe = xte[:500]
        tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        model.eval()
        with model.mode_ctx("max"):
            before = model(probe)
        default = {g: imp.argsort(descending=True, stable=True) for g, imp in model.channel_importance().items()}
        perms = sort_channels(model)
        with model.mode_ctx("max"):
            after = model(probe)
        model.train()
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
        moved = sum(int((perm[: model.base_sizes[g]] >= model.base_sizes[g]).sum()) for g, perm in perms.items())
        entry = {
            "epoch": epoch,
            "groups": len(perms),
            "base_channels": sum(model.base_sizes[g] for g in perms),
            "moved_into_base": moved,
            # groups whose order this sort changed
            "sorted_groups": sum(not is_identity(p) for p in perms.values()),
            # groups whose permutation differs from the default sort (sum, all groups); a
            # group outside the scope counts if the default sort would have moved it
            "diff_from_default": sum(
                not (is_identity(p) if g not in perms else torch.equal(perms[g], p)) for g, p in default.items()
            ),
            "max_abs_diff": (after - before).abs().max().item(),
            "max_abs_logit": before.abs().max().item(),
        }
        sort_log.append(entry)
        print("sort:", entry, flush=True)

    probe = GradProbe(model, args.alpha, backward) if isinstance(model, NetAugModel) and args.grad_log_every else None
    grad_log = []

    epoch_loss = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        if isinstance(model, NetAugModel) and sorting and epoch and epoch < aug_until and epoch % args.sort_every == 0:
            sort_and_log(epoch)
        perm = torch.randperm(xtr.shape[0], device=device, generator=gen)
        loss_sum = torch.zeros((), device=device)
        for s in range(steps_per_epoch):
            idx = perm[s * args.bs : (s + 1) * args.bs]
            xb, yb = augment(xtr[idx], gen), ytr[idx]
            if isinstance(model, NetAugModel):
                logged = probe is not None and s % args.grad_log_every == 0
                loss, _ = netaug_step(
                    model,
                    xb,
                    yb,
                    crit,
                    alpha=args.alpha,
                    augment=epoch < aug_until,
                    generator=rng,
                    granularity=args.granularity,
                    forward=fwd,
                    backward=probe.backward if logged else backward,
                )
                if logged:
                    probe.finish_step()
            else:
                loss = crit(fwd(model, xb), yb)
                backward(loss)
                loss = loss.detach()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            loss_sum += loss
        epoch_loss.append(loss_sum.item() / steps_per_epoch)  # base-model training loss
        if probe is not None:
            grad_log.append({"epoch": epoch, **probe.epoch_summary()})
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            g = ""
            if grad_log and "cos" in grad_log[-1]:
                g = f"  grad cos {grad_log[-1]['cos']:.3f} ratio {grad_log[-1]['ratio']:.2f}"
            print(
                f"epoch {epoch + 1}/{args.epochs}  base loss {epoch_loss[-1]:.3f}  {time.time() - t0:.0f}s{g}",
                flush=True,
            )
    train_minutes = (time.time() - t0) / 60

    # Every evaluated model gets the same BN calibration: exact averages over 20 clean
    # training batches, in float32.
    calib = [xtr[k : k + 500] for k in range(0, 10000, 500)]

    def eval_at(sizes, depths=None) -> dict:
        # All widths share the BN buffers, so calibrate each width on its own copy.
        assert isinstance(model, NetAugModel)
        depths = dict(depths or model.base_depths)
        sub = copy.deepcopy(model)
        sub.set_state((sizes, depths, "custom"))
        recalibrate_bn(sub, calib, sizes=sizes, depths=depths)  # restores the same state
        # The exported model is plain PyTorch (no torchnetaug layers) at exactly this width.
        exported = sub.export(sizes, depths)
        sub.eval(), exported.eval()
        with torch.no_grad():
            diff = (sub(xte[:1000]) - exported(xte[:1000])).abs().max().item()
        out = {
            "params": sub.num_active_params(),
            "test_acc": evaluate(sub, xte, yte),
            "train_acc": evaluate(sub, xtr[:10000], ytr[:10000]),
            "export_params": sum(q.numel() for q in exported.parameters()),
            "export_test_acc": evaluate(exported, xte, yte),
            "export_max_abs_diff": diff,
        }
        del sub, exported
        return out

    width_acc, random_acc = {}, []
    if isinstance(model, NetAugModel):
        # Global widths: every free group scaled by the same factor. x1.5 and x2.5 are
        # never sampled in training (each group samples from x1, x2, x3).
        for m in args.eval_widths:
            model.set_width(m)
            width_acc[str(m)] = eval_at(list(model.active_sizes))
        # Per-group random widths: the distribution actually sampled in training.
        eval_rng = random.Random(10_000 + args.seed)
        for _ in range(args.eval_random):
            model.set_random(eval_rng)
            random_acc.append(eval_at(list(model.active_sizes), model.active_depths))
        if args.depth:  # base width, every extra block on
            width_acc["depth_max"] = eval_at(list(model.base_sizes), model.max_depths)
        model.set_base()

    recalibrate_bn(model, calib)
    final = model.export() if isinstance(model, NetAugModel) else model
    test_acc = evaluate(final, xte, yte)
    train_acc = evaluate(final, xtr[:10000], ytr[:10000])
    rec = {
        "method": args.method,
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "width_mults": args.width_mults,
        "alpha": args.alpha,
        "stop_aug": args.stop_aug,
        "sort_every": key[4],
        "width": args.width,
        "depth": args.depth,
        "lr": args.lr,
        "wd": args.wd,
        "amp": args.amp,
        "init": args.init,
        "warmup": args.warmup,
        "granularity": args.granularity,
        "sort_scope": args.sort_scope,
        "sort_agg": args.sort_agg,
        "sort_opt": args.sort_opt,
        "params": n_base,
        "test_acc": test_acc,
        "train_acc": train_acc,
        "width_acc": width_acc,
        "random_acc": random_acc,
        "sort_log": sort_log,
        "epoch_loss": epoch_loss,
        "grad_log": grad_log,
        "minutes": train_minutes,
    }
    print(json.dumps(rec))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
