"""Depth augmentation: extra copies of chosen blocks that only augmented models run.

Checked for a residual CNN (TinyResNet, torchvision ResNet-18, MobileNetV2) and a plain
(non-residual) conv stack:

* base mode and ``export()`` compute exactly the original model;
* a residual extra block starts as the identity;
* channel sorting leaves the output unchanged at every width *and* depth, base included
  (this needs the extra blocks' outputs to share the chosen block's channel group);
* ``export(sizes, depths)`` is a plain PyTorch model that computes the active network;
* depth state survives ``mode_ctx``, ``export`` and ``recalibrate_bn``;
* parameter counts and error messages.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn as nn
import torchvision
from conftest import assert_sorted, randomize
from test_widths import BASE_WIDTHS, TinyResNet

from torchnetaug import DepthStack, NetAugModel, netaug_step, recalibrate_bn


class Block(nn.Sequential):
    """Conv-BN-ReLU without a residual connection."""

    def __init__(self, c: int) -> None:
        super().__init__(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())


class PlainStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.body = nn.Sequential(Block(8), Block(8))
        self.head = nn.Sequential(nn.Conv2d(8, 12, 1), nn.BatchNorm2d(12), nn.ReLU())
        self.fc = nn.Linear(12, 5)

    def forward(self, x):
        x = self.head(self.body(self.stem(x)))
        return self.fc(x.mean((2, 3)))


CASES = {
    "tiny_resnet": (lambda: TinyResNet(BASE_WIDTHS),
                    {"layers.1": 1, "layers.3": 1, "layers.5": 2, "layers.7": 1}),
    "resnet18": (lambda: torchvision.models.resnet18(num_classes=10), {"layer1": 1, "layer3": 2}),
    "mobilenet_v2": (lambda: torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10),
                     {"features.3": 1, "features.6": 2}),
    "plain_stack": (PlainStack, {"body": 2}),
}
RESIDUAL = {"tiny_resnet", "resnet18", "mobilenet_v2"}


def _x():
    return torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(1))


def _make(case, mults=(1.0, 2.0, 3.0)):
    factory, depth = CASES[case]
    torch.manual_seed(0)
    model = randomize(factory()).eval()
    netaug = NetAugModel(model, _x(), width_mults=mults, depth=depth).eval()
    return model, netaug


def _no_package_modules(model: nn.Module) -> bool:
    return not any(type(m).__module__.startswith("torchnetaug") for m in model.modules())


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("case", sorted(CASES))
def test_base_mode_equals_original(case, train):
    model, netaug = _make(case)
    model.train(train), netaug.train(train)
    x = _x()
    with torch.no_grad():
        torch.manual_seed(0)  # dropout (MobileNetV2) draws the same mask in both calls
        expected = model(x)
        torch.manual_seed(0)
        torch.testing.assert_close(netaug(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("case", sorted(CASES))
def test_base_export_is_the_original_model(case):
    model, netaug = _make(case)
    netaug.set_max()  # export() must not depend on the current mode
    exported = netaug.export().eval()
    assert type(exported) is type(model)
    assert list(exported.state_dict()) == list(model.state_dict())
    assert _no_package_modules(exported)
    x = _x()
    with torch.no_grad():
        torch.testing.assert_close(exported(x), model(x), rtol=0, atol=0)
    assert netaug.mode == "max" and netaug.active_depths == netaug.max_depths


@pytest.mark.parametrize("case", sorted(RESIDUAL))
def test_residual_extra_blocks_start_as_identity(case):
    factory, depth = CASES[case]
    torch.manual_seed(0)
    model = factory().eval()  # default init: extra blocks are new, base weights untouched
    netaug = NetAugModel(model, _x(), depth=depth).eval()
    netaug.set_depths(netaug.max_depths)
    with torch.no_grad():
        torch.testing.assert_close(netaug(_x()), model(_x()), rtol=1e-6, atol=1e-6)


def test_sequential_subclass_is_one_block():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), Block(8), nn.Flatten(), nn.LazyLinear(5))
    net(_x())  # materialize the lazy layer
    netaug = NetAugModel(net, _x(), depth={"1": 1})
    assert isinstance(netaug.model[1], DepthStack) and netaug.max_depths == {"1": 1}


def test_plain_extra_blocks_are_not_zeroed():
    torch.manual_seed(0)
    netaug = NetAugModel(PlainStack(), _x(), depth={"body": 1})
    extra = netaug.model.body[1].extra[0]
    assert isinstance(netaug.model.body[1], DepthStack)
    assert extra[1].weight.abs().min() > 0  # zeroing would cut the signal of a plain block


def _max_width_outputs(netaug, x):
    """Output at max width for every depth combination (all 0, all max, and mixes)."""
    combos = [dict.fromkeys(netaug.max_depths, 0), dict(netaug.max_depths)]
    rng = random.Random(3)
    combos += [{p: rng.randint(0, k) for p, k in netaug.max_depths.items()} for _ in range(4)]
    res = []
    with torch.no_grad():
        for depths in combos:
            netaug.set_state((netaug.max_sizes, depths, "max"))
            res.append(netaug(x))
    netaug.set_base()
    return res


@pytest.mark.parametrize("case", sorted(CASES))
def test_sorting_keeps_the_max_width_network_at_every_depth(case):
    """Sorting permutes channels; at max width every channel is kept, so nothing changes.

    This holds at depth 0 too only if the layer after a depth point reads the same channel
    order whether the extra blocks run or not, i.e. the extra blocks' outputs are in the
    chosen block's channel group. (Smaller widths keep other channels after sorting: that
    is the purpose of sorting.)
    """
    _, netaug = _make(case)
    with torch.no_grad():  # make the extra blocks non-trivial so a wrong permutation shows
        for p in netaug._depth_paths:
            randomize(netaug.model.get_submodule(p).extra, seed=7)
    x = _x()
    before = _max_width_outputs(netaug, x)
    perms = netaug.sort_channels()
    after = _max_width_outputs(netaug, x)
    for b, a in zip(before, after):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    assert any((p != torch.arange(p.numel())).any() for p in perms.values())
    assert_sorted(netaug)
    # the sorted base network is still exactly what export() returns
    exported = netaug.export().eval()
    with torch.no_grad():
        torch.testing.assert_close(exported(x), netaug(x), rtol=0, atol=0)


@pytest.mark.parametrize("case", sorted(CASES))
def test_export_at_width_and_depth(case):
    _, netaug = _make(case)
    with torch.no_grad():
        for p in netaug._depth_paths:
            randomize(netaug.model.get_submodule(p).extra, seed=7)
    rng, x = random.Random(5), _x()
    for _ in range(3):
        netaug.set_random(rng)
        sizes, depths, _ = netaug.state()
        exported = netaug.export(sizes, depths).eval()
        assert _no_package_modules(exported)
        assert sum(p.numel() for p in exported.parameters()) == netaug.num_active_params()
        with torch.no_grad():
            torch.testing.assert_close(exported(x), netaug(x), rtol=0, atol=0)
        assert netaug.state() == (sizes, depths, "random")  # export restores the state


def test_state_is_restored():
    _, netaug = _make("tiny_resnet")
    netaug.set_depths({"layers.5": 1})
    saved = netaug.state()
    with netaug.mode_ctx("max"):
        assert netaug.active_depths == netaug.max_depths
    assert netaug.state() == saved
    recalibrate_bn(netaug, [_x()])
    assert netaug.state() == saved
    netaug.set_base()
    assert netaug.is_base and all(netaug.model.get_submodule(p).active == 0 for p in netaug._depth_paths)


def test_recalibrate_bn_at_depth_updates_extra_blocks():
    _, netaug = _make("tiny_resnet")
    bn = netaug.model.get_submodule("layers.5").extra[1].bn1
    before = bn.running_mean.clone()
    with torch.no_grad():
        bn.running_mean.normal_()  # e.g. statistics learned with update_bn_in_aug=True
    before = bn.running_mean.clone()
    recalibrate_bn(netaug, [_x()])  # base depth: the extra block does not run, keeps its stats
    torch.testing.assert_close(bn.running_mean, before, rtol=0, atol=0)
    recalibrate_bn(netaug, [_x()], depths=netaug.max_depths)
    assert not torch.equal(bn.running_mean, before)


def test_netaug_step_trains_extra_blocks_and_keeps_base_bn():
    _, netaug = _make("tiny_resnet")
    netaug.train()
    extra = netaug.model.get_submodule("layers.7").extra[0]
    seed = next(s for s in range(100)  # a seed whose sample runs the layers.7 extra block
                if (netaug.set_random(random.Random(s)), netaug.active_depths["layers.7"])[1] == 1)
    netaug.set_base()
    extra_bn = extra.bn1
    stats = extra_bn.running_mean.clone()
    x, y = _x(), torch.tensor([1, 2])
    netaug_step(netaug, x, y, nn.functional.cross_entropy, generator=random.Random(seed))
    # the block starts as the identity (bn2 weight 0): its last BN gets the first gradient
    assert extra.bn2.weight.grad is not None and extra.bn2.weight.grad.abs().sum() > 0
    assert netaug.is_base
    # the augmented pass does not touch BN statistics (update_bn_in_aug=False)
    torch.testing.assert_close(extra_bn.running_mean, stats, rtol=0, atol=0)


def test_set_random_samples_every_depth():
    _, netaug = _make("tiny_resnet")
    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        netaug.set_random(rng)
        seen.add(netaug.active_depths["layers.5"])
    assert seen == {0, 1, 2}


def test_set_random_without_depth_samples_like_before():
    torch.manual_seed(0)
    netaug = NetAugModel(TinyResNet(BASE_WIDTHS), _x())
    a, b = random.Random(0), random.Random(0)
    netaug.set_random(a)
    expected = [b.choice(c) for c in netaug.choices]
    assert netaug.active_sizes == expected and a.random() == b.random()


class Twice(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        return self.conv(self.conv(x))


@pytest.mark.parametrize(
    "factory, depth, match",
    [
        (lambda: TinyResNet(BASE_WIDTHS), {"layers.2": 1}, "changes the shape"),
        (lambda: TinyResNet(BASE_WIDTHS), {"nope": 1}, "no module named"),
        (lambda: TinyResNet(BASE_WIDTHS), {"layers.1": 0}, "at least 1"),
        (lambda: TinyResNet(BASE_WIDTHS), {"layers": 1, "layers.7": 1}, "given twice"),
        (lambda: TinyResNet(BASE_WIDTHS), {"layers.1": 1, "layers.1.conv1": 1}, "inside"),
        (Twice, {"conv": 1}, "exactly once"),
        (lambda: nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU()), {"1": 1}, "no parameters"),
    ],
)
def test_depth_errors(factory, depth, match):
    with pytest.raises(ValueError, match=match):
        NetAugModel(factory(), _x(), depth=depth)


def test_depth_needs_tracing():
    model = TinyResNet(BASE_WIDTHS)
    netaug = NetAugModel(model, _x())
    with pytest.raises(ValueError, match="pre-built graph"):
        NetAugModel(model, graph=netaug.graph, depth={"layers.1": 1})


def test_bare_layer_as_depth_point_exports():
    """A depth point may be a single layer (not a block): export at width and depth works."""
    torch.manual_seed(0)
    model = randomize(nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))).eval()
    netaug = NetAugModel(model, _x(), depth={"1": 1}).eval()
    x = _x()
    for state in ("base", "max"):
        getattr(netaug, f"set_{state}")()
        sizes, depths, _ = netaug.state()
        exported = netaug.export(sizes, depths).eval()
        assert _no_package_modules(exported)
        with torch.no_grad():
            torch.testing.assert_close(exported(x), netaug(x), rtol=0, atol=0)


def test_depth_keeps_per_module_train_flags():
    """Frozen (eval-mode) submodules stay frozen when depth= is used, as without it."""
    model = TinyResNet(BASE_WIDTHS).train()
    frozen = ["layers.0.bn1", "layers.1.bn1", "layers.2.bn2"]  # layers.1 is the depth point
    for name in frozen:
        model.get_submodule(name).eval()
    netaug = NetAugModel(model, _x(), depth={"layers.1": 1})
    for name in ["layers.0.bn1", "layers.1.block.bn1", "layers.1.extra.0.bn1", "layers.2.bn2"]:
        assert not netaug.model.get_submodule(name).training, name
    assert netaug.model.get_submodule("layers.1.block.bn2").training
    assert netaug.model.get_submodule("layers.3").training


def test_recalibrate_bn_restores_state_and_flags_on_error():
    _, netaug = _make("tiny_resnet")
    netaug.train()
    netaug.model.get_submodule("layers.0.bn1").eval()
    netaug.set_max()
    saved = netaug.state()
    bn = netaug.model.get_submodule("layers.0.bn1")
    stats = bn.running_mean.clone()
    with pytest.raises(ValueError):
        recalibrate_bn(netaug, [_x()], depths={"no.such.block": 1})
    assert netaug.state() == saved
    torch.testing.assert_close(bn.running_mean, stats, rtol=0, atol=0)  # nothing was reset
    recalibrate_bn(netaug, [_x()])
    assert netaug.state() == saved
    assert not bn.training and netaug.model.get_submodule("layers.1").training


def test_module_conversions_work():
    """NetAugModel must not shadow nn.Module._apply: .to(), .float(), .double() keep working."""
    _, netaug = _make("tiny_resnet")
    netaug.set_depths({"layers.5": 1})
    saved = netaug.state()
    netaug.float()
    assert all(p.dtype == torch.float32 for p in netaug.parameters())
    netaug.to(torch.float64)
    assert all(p.dtype == torch.float64 for p in netaug.parameters())
    assert netaug.state() == saved
    if torch.cuda.is_available():
        netaug.cuda()
        assert all(p.is_cuda for p in netaug.parameters())
        netaug.cpu()
