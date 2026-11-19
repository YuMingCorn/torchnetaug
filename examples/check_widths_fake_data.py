"""Check the dynamic widths of TinyResNet on small synthetic data, before any real training.

    uv run python examples/check_widths_fake_data.py

The data are 10 classes; each class is a fixed random 3x32x32 pattern plus strong noise,
so the task is learnable but not trivial. The script:

1. wraps TinyResNet (the model of the CIFAR experiment) with NetAugModel;
2. at each global width x1.0 ... x3.0, builds a plain TinyResNet directly at that width,
   copies the active weights into it, and compares parameter count and output;
3. trains with ``netaug_step`` and channel sorting, then repeats step 2 and reports the
   accuracy of every width after BN recalibration.

Every check prints PASS or FAIL; the script exits with code 1 on any FAIL.
"""

from __future__ import annotations

import copy
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from cifar_netaug import TinyResNet

from torchnetaug import NetAugModel, netaug_step, recalibrate_bn

BASE = (8, 16, 32, 64)
MULTS = (1.0, 1.5, 2.0, 2.5, 3.0)
failures = []


def check(ok: bool, what: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    if not ok:
        failures.append(what)


def make_data(n: int, templates: torch.Tensor, gen: torch.Generator, noise: float = 6.0):
    y = torch.randint(0, templates.shape[0], (n,), generator=gen)
    x = templates[y] + noise * torch.randn(n, 3, 32, 32, generator=gen)
    return x, y


def standalone(netaug: NetAugModel, mult: float) -> nn.Module:
    """Plain TinyResNet at ``mult`` x width holding the active (prefix) weights."""
    ref = TinyResNet(tuple(max(1, round(w * mult)) for w in BASE))
    state = {}
    for name, t in ref.state_dict().items():
        full = netaug.model.state_dict()[name]
        state[name] = full[tuple(slice(0, s) for s in t.shape)].clone() if t.ndim else full.clone()
    ref.load_state_dict(state)
    return ref.to(next(netaug.parameters()).device)


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return (model(x).argmax(1) == y).float().mean().item()


def compare_widths(netaug: NetAugModel, x_check: torch.Tensor, calib, x_test=None, y_test=None) -> None:
    print("  width | params (dynamic) | params (plain model) | max |out diff| | test acc")
    for m in MULTS:
        sub = copy.deepcopy(netaug)
        sub.set_width(m)
        if calib is not None:
            recalibrate_bn(sub, calib, sizes=list(sub.active_sizes))
        ref = standalone(sub, m)
        n_dyn, n_ref = sub.num_active_params(), sum(p.numel() for p in ref.parameters())
        sub.eval(), ref.eval()
        with torch.no_grad():
            out, out_ref = sub(x_check), ref(x_check)
        diff = (out - out_ref).abs().max().item()
        scale = out_ref.abs().max().item()
        acc = "" if x_test is None else f"{accuracy(sub, x_test, y_test) * 100:6.2f}%"
        print(f"  x{m:<4} | {n_dyn:16,} | {n_ref:20,} | {diff:.2e} (|logit| max {scale:.1f}) | {acc}")
        check(n_dyn == n_ref, f"x{m}: parameter count equals the plain model")
        check(diff <= 1e-4 * max(1.0, scale), f"x{m}: output equals the plain model")


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator().manual_seed(0)
    templates = torch.randn(10, 3, 32, 32, generator=gen)
    x_tr, y_tr = (t.to(device) for t in make_data(2048, templates, gen))
    x_te, y_te = (t.to(device) for t in make_data(1000, templates, gen))

    netaug = NetAugModel(TinyResNet(BASE).to(device), x_tr[:2], width_mults=(1.0, 2.0, 3.0))
    print(netaug.graph.summary().splitlines()[0])

    print("\n1) before training: every width equals a plain model of that width")
    compare_widths(netaug, x_te[:64], calib=None)

    print("\n2) NetAug training on the synthetic data (sorting every 100 steps)")
    opt = torch.optim.SGD(netaug.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    rng = random.Random(0)
    steps, bs = 400, 128
    first = last = None
    for step in range(1, steps + 1):
        netaug.train()
        idx = torch.randint(0, x_tr.shape[0], (bs,), device=device)
        base_loss, aug_loss = netaug_step(netaug, x_tr[idx], y_tr[idx], F.cross_entropy, generator=rng)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 1:
            first = (base_loss.item(), aug_loss.item())
        if step % 100 == 0:
            last = (base_loss.item(), aug_loss.item())
            print(f"  step {step}: base loss {last[0]:.3f}, augmented loss {last[1]:.3f}")
            if step < steps:
                netaug.eval()
                with netaug.mode_ctx("max"), torch.no_grad():
                    before = netaug(x_te[:64])
                    netaug.sort_channels(optimizer=opt)
                    after = netaug(x_te[:64])
                d = (before - after).abs().max().item()
                check(d <= 1e-4 * max(1.0, before.abs().max().item()),
                      f"step {step}: sorting keeps the max-width output (max diff {d:.1e})")
    check(last[0] < first[0] and last[1] < first[1], "base and augmented losses go down")

    print("\n3) after training: each width, BN recalibrated, against a plain model of that width")
    calib = [x_tr[k:k + 128] for k in range(0, 1024, 128)]
    compare_widths(netaug, x_te[:64], calib=calib, x_test=x_te, y_test=y_te)

    netaug.set_base()
    recalibrate_bn(netaug, calib)
    exported = netaug.export()
    acc_base, acc_export = accuracy(netaug, x_te, y_te), accuracy(exported, x_te, y_te)
    check(acc_base == acc_export, f"export() keeps the base accuracy ({acc_export * 100:.2f}%)")
    check(sum(p.numel() for p in exported.parameters()) == 176_402, "export() has the base size (176,402)")

    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAIL'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
