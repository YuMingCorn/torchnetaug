"""Compare plain training with NetAug training of a tiny model on CIFAR.

Example:
    uv run python examples/cifar_netaug.py --method baseline --seed 0
    uv run python examples/cifar_netaug.py --method netaug --seed 0
    uv run python examples/cifar_netaug.py --method netaug --seed 0 --sort-every 10
    uv run python examples/cifar_netaug.py --method netaug --seed 0 --depth 1 --width-mults 1.0

Each run appends one JSON line to ``--out`` (default ``runs/cifar_results.jsonl``).
The whole dataset lives on the GPU and augmentation (random crop + flip) runs there too,
so no DataLoader workers are needed.

Training runs the forward pass in bfloat16; the loss, BN calibration and all
evaluation run in float32.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchnetaug import NetAugModel, netaug_step, recalibrate_bn

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
    out = padded[torch.arange(n, device=x.device).view(n, 1, 1, 1),
                 torch.arange(3, device=x.device).view(1, 3, 1, 1), rows, cols]
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
        correct += (model(x[k:k + bs]).argmax(1) == y[k:k + bs]).sum().item()
    return correct / x.shape[0]


def run_key(r: dict) -> tuple:
    return (r["dataset"], r["model"], r["epochs"], r["method"], r["sort_every"], r["seed"],
            tuple(r["width_mults"]), r["alpha"], r["stop_aug"], r.get("width", 1.0), r.get("depth", 0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["baseline", "netaug"], required=True)
    p.add_argument("--model", default="resnet_tiny", choices=["mbv2_035", "mbv2_050", "resnet_tiny"])
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width-mults", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--stop-aug", type=float, default=0.0,
                   help="fraction of final epochs trained without augmentation")
    p.add_argument("--sort-every", type=int, default=0, help="sort channels every N epochs (0 = never)")
    p.add_argument("--eval-widths", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0],
                   help="NetAug only: also report accuracy at these global width multipliers")
    p.add_argument("--eval-random", type=int, default=3,
                   help="NetAug only: also report accuracy of this many per-group random widths")
    p.add_argument("--width", type=float, default=1.0,
                   help="baseline only: train a plain TinyResNet at this width (reference for the exported widths)")
    p.add_argument("--depth", type=int, default=0,
                   help="depth augmentation: extra blocks after each TinyResNet stage (netaug only)")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/cifar_results.jsonl")
    p.add_argument("--skip-existing", action="store_true", help="skip if --out already has this run")
    args = p.parse_args()

    key = run_key({"dataset": args.dataset, "model": args.model, "epochs": args.epochs,
                   "method": args.method, "sort_every": args.sort_every if args.method == "netaug" else 0,
                   "seed": args.seed, "width_mults": args.width_mults, "alpha": args.alpha,
                   "stop_aug": args.stop_aug, "width": args.width, "depth": args.depth})
    if args.depth and (args.method != "netaug" or args.model != "resnet_tiny"):
        p.error("--depth is only for --method netaug --model resnet_tiny")
    if args.width != 1.0 and (args.method != "baseline" or args.model != "resnet_tiny"):
        p.error("--width is only for --method baseline --model resnet_tiny")
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
        model: nn.Module = NetAugModel(base, xtr[:2], width_mults=args.width_mults, depth=depth)
        print(model.graph.summary())
    else:
        model = base

    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, momentum=0.9, nesterov=True)
    steps_per_epoch = xtr.shape[0] // args.bs
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    aug_until = int(round(args.epochs * (1 - args.stop_aug)))

    def crit(out, y):  # float32 loss even though the forward pass runs in bf16
        return F.cross_entropy(out.float(), y)

    def fwd(m, inp):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(inp)

    sort_log = []

    @torch.no_grad()
    def sort_and_log(epoch: int) -> None:
        """Sort channels and record evidence that the sort (a) moved channels across the
        base boundary and (b) kept the max-width function on the real trained weights."""
        probe = xte[:500]
        tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        model.eval()
        with model.mode_ctx("max"):
            before = model(probe)
        perms = model.sort_channels(optimizer=opt)
        with model.mode_ctx("max"):
            after = model(probe)
        model.train()
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
        moved = sum(int((perm[: model.base_sizes[g]] >= model.base_sizes[g]).sum()) for g, perm in perms.items())
        entry = {"epoch": epoch, "groups": len(perms),
                 "base_channels": sum(model.base_sizes[g] for g in perms),
                 "moved_into_base": moved,
                 "max_abs_diff": (after - before).abs().max().item(),
                 "max_abs_logit": before.abs().max().item()}
        sort_log.append(entry)
        print("sort:", entry, flush=True)

    epoch_loss = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        if args.method == "netaug" and args.sort_every and epoch and epoch % args.sort_every == 0:
            sort_and_log(epoch)
        perm = torch.randperm(xtr.shape[0], device=device, generator=gen)
        loss_sum = torch.zeros((), device=device)
        for s in range(steps_per_epoch):
            idx = perm[s * args.bs:(s + 1) * args.bs]
            xb, yb = augment(xtr[idx], gen), ytr[idx]
            if args.method == "netaug":
                loss, _ = netaug_step(model, xb, yb, crit, alpha=args.alpha, augment=epoch < aug_until,
                                      generator=rng, forward=fwd)
            else:
                loss = crit(fwd(model, xb), yb)
                loss.backward()
                loss = loss.detach()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            loss_sum += loss
        epoch_loss.append(loss_sum.item() / steps_per_epoch)  # base-model training loss
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(f"epoch {epoch + 1}/{args.epochs}  base loss {epoch_loss[-1]:.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    train_minutes = (time.time() - t0) / 60

    # Every evaluated model gets the same BN calibration: exact averages over 20 clean
    # training batches, in float32.
    calib = [xtr[k:k + 500] for k in range(0, 10000, 500)]

    def eval_at(sizes, depths=None) -> dict:
        # All widths share the BN buffers, so calibrate each width on its own copy.
        depths = dict(depths or model.base_depths)
        sub = copy.deepcopy(model)
        sub.set_state((sizes, depths, "custom"))
        recalibrate_bn(sub, calib, sizes=sizes, depths=depths)  # restores the same state
        # The exported model is plain PyTorch (no torchnetaug layers) at exactly this width.
        exported = sub.export(sizes, depths)
        sub.eval(), exported.eval()
        with torch.no_grad():
            diff = (sub(xte[:1000]) - exported(xte[:1000])).abs().max().item()
        out = {"params": sub.num_active_params(), "test_acc": evaluate(sub, xte, yte),
               "train_acc": evaluate(sub, xtr[:10000], ytr[:10000]),
               "export_params": sum(q.numel() for q in exported.parameters()),
               "export_test_acc": evaluate(exported, xte, yte), "export_max_abs_diff": diff}
        del sub, exported
        return out

    width_acc, random_acc = {}, []
    if args.method == "netaug":
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
    final = model.export() if args.method == "netaug" else model
    test_acc = evaluate(final, xte, yte)
    train_acc = evaluate(final, xtr[:10000], ytr[:10000])
    rec = {"method": args.method, "model": args.model, "dataset": args.dataset, "seed": args.seed,
           "epochs": args.epochs, "width_mults": args.width_mults, "alpha": args.alpha,
           "stop_aug": args.stop_aug, "sort_every": key[4], "width": args.width, "depth": args.depth,
           "params": n_base,
           "test_acc": test_acc, "train_acc": train_acc, "width_acc": width_acc,
           "random_acc": random_acc, "sort_log": sort_log, "epoch_loss": epoch_loss,
           "minutes": train_minutes}
    print(json.dumps(rec))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
