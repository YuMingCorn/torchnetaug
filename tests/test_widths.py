"""Is each width really the network it claims to be?

Two levels, both on small random data:

* Layer level: a dynamic layer at some active size must compute exactly what a plain
  PyTorch layer of that size computes, given the same (sliced) weights.
* Model level: ``NetAugModel`` at a global width must equal a model built directly at
  that width from the model's own class: same parameter count, same weight shape in
  every layer, same output in eval and train mode, same BN statistics update.

The reference for the model level is built independently of the library: its shapes
come from the model's constructor, not from the channel graph.
"""

from __future__ import annotations

import copy
import importlib.util
import itertools
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from conftest import example_input, randomize
from torch import nn

from torchnetaug import DynamicBatchNorm, DynamicConv2d, DynamicLinear, NetAugModel

ROOT = Path(__file__).resolve().parents[1]


def _load_experiment():
    spec = importlib.util.spec_from_file_location("cifar_netaug", ROOT / "examples" / "cifar_netaug.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TinyResNet = _load_experiment().TinyResNet
BASE_WIDTHS = (8, 16, 32, 64)
MULTS = (1.0, 1.5, 2.0, 2.5, 3.0)


def _index(segments):
    """Flat channel index of a segment spec, written out by hand (independent of the library)."""
    return [off + k for off, act in segments for k in range(act)]


# ----------------------------------------------------------------------------------
# layer level
# ----------------------------------------------------------------------------------
SEGMENTS = [((0, 5),), ((0, 12),), ((0, 2), (6, 3)), ((0, 4), (6, 6), (12, 1))]


@pytest.mark.parametrize("segments", SEGMENTS)
@pytest.mark.parametrize("out", [1, 7, 16])
def test_dense_conv_equals_plain_conv(segments, out):
    torch.manual_seed(0)
    dyn = DynamicConv2d(13, 16, 3, padding=1, bias=True)
    dyn.set_in_segments(segments)
    dyn.active_out = out
    idx = torch.tensor(_index(segments))
    plain = nn.Conv2d(len(idx), out, 3, padding=1, bias=True)
    assert plain.bias is not None and dyn.bias is not None
    with torch.no_grad():
        plain.weight.copy_(dyn.weight[:out][:, idx])
        plain.bias.copy_(dyn.bias[:out])
    x = torch.randn(2, len(idx), 6, 6)
    assert dyn.active_weight().shape == plain.weight.shape
    torch.testing.assert_close(dyn(x), plain(x), rtol=0, atol=0)


@pytest.mark.parametrize("segments", SEGMENTS)
def test_depthwise_conv_equals_plain_conv(segments):
    torch.manual_seed(0)
    dyn = DynamicConv2d(13, 13, 3, padding=1, groups=13, bias=True)
    dyn.set_in_segments(segments)
    idx = torch.tensor(_index(segments))
    n = len(idx)
    plain = nn.Conv2d(n, n, 3, padding=1, groups=n, bias=True)
    assert plain.bias is not None and dyn.bias is not None
    with torch.no_grad():
        plain.weight.copy_(dyn.weight[idx])
        plain.bias.copy_(dyn.bias[idx])
    x = torch.randn(2, n, 6, 6)
    assert dyn.active_weight().shape == plain.weight.shape
    torch.testing.assert_close(dyn(x), plain(x), rtol=0, atol=0)


@pytest.mark.parametrize("segments", SEGMENTS)
@pytest.mark.parametrize("out", [1, 9])
def test_linear_equals_plain_linear(segments, out):
    torch.manual_seed(0)
    dyn = DynamicLinear(13, 9)
    dyn.set_in_segments(segments)
    dyn.active_out = out
    idx = torch.tensor(_index(segments))
    plain = nn.Linear(len(idx), out)
    with torch.no_grad():
        plain.weight.copy_(dyn.weight[:out][:, idx])
        plain.bias.copy_(dyn.bias[:out])
    x = torch.randn(4, len(idx))
    torch.testing.assert_close(dyn(x), plain(x), rtol=0, atol=0)


@pytest.mark.parametrize("segments", SEGMENTS)
@pytest.mark.parametrize("momentum", [0.1, None])
def test_batchnorm_equals_plain_batchnorm(segments, momentum):
    torch.manual_seed(0)
    dyn = DynamicBatchNorm(13, momentum=momentum)
    randomize(dyn)
    dyn.set_in_segments(segments)
    idx = torch.tensor(_index(segments))
    n = len(idx)
    plain = nn.BatchNorm2d(n, momentum=momentum)
    assert dyn.running_mean is not None and dyn.running_var is not None and dyn.num_batches_tracked is not None
    assert plain.running_mean is not None and plain.running_var is not None
    with torch.no_grad():
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(plain, attr).copy_(getattr(dyn, attr)[idx])
    untouched = [i for i in range(13) if i not in set(idx.tolist())]
    before = dyn.running_mean.clone(), dyn.running_var.clone()

    # train mode, statistics updated: output and the selected running stats must match
    x = torch.randn(4, n, 5, 5)
    dyn.train(), plain.train()
    dyn.update_stats = True
    torch.testing.assert_close(dyn(x), plain(x))
    torch.testing.assert_close(dyn.running_mean[idx], plain.running_mean)
    torch.testing.assert_close(dyn.running_var[idx], plain.running_var)
    # channels outside the active set keep their statistics
    torch.testing.assert_close(dyn.running_mean[untouched], before[0][untouched], rtol=0, atol=0)
    torch.testing.assert_close(dyn.running_var[untouched], before[1][untouched], rtol=0, atol=0)

    # eval mode uses the running statistics
    dyn.eval(), plain.eval()
    torch.testing.assert_close(dyn(x), plain(x))

    # train mode without updates: batch statistics, buffers unchanged
    dyn.train(), plain.train()
    dyn.update_stats = False
    snap = dyn.running_mean.clone(), dyn.running_var.clone(), dyn.num_batches_tracked.clone()
    plain_copy = copy.deepcopy(plain)
    torch.testing.assert_close(dyn(x), plain_copy(x))
    assert torch.equal(dyn.running_mean, snap[0]) and torch.equal(dyn.running_var, snap[1])
    assert torch.equal(dyn.num_batches_tracked, snap[2])


# ----------------------------------------------------------------------------------
# model level: TinyResNet (the model used in the CIFAR experiment)
# ----------------------------------------------------------------------------------
def _standalone_at(netaug: NetAugModel, mult: float) -> nn.Module:
    """A TinyResNet built directly at ``mult`` x width, holding the active weights."""
    widths = tuple(max(1, round(w * mult)) for w in BASE_WIDTHS)
    ref = TinyResNet(widths)
    dyn = dict(netaug.dynamic_layers())
    state = {}
    for name, t in ref.state_dict().items():
        layer_name, attr = name.rsplit(".", 1)
        src = netaug.model.get_submodule(layer_name)
        full = getattr(src, attr)
        if layer_name in dyn and attr != "num_batches_tracked":
            # plain prefix slice to the reference shape; the slicing below never uses
            # the library's segment logic
            full = full[tuple(slice(0, s) for s in t.shape)]
        assert full.shape == t.shape, (name, full.shape, t.shape)
        state[name] = full.detach().clone()
    ref.load_state_dict(state)
    return ref


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = TinyResNet(BASE_WIDTHS)
    randomize(model)
    return NetAugModel(model, torch.randn(2, 3, 32, 32), width_mults=(1.0, 2.0, 3.0))


def test_tiny_resnet_is_fully_augmentable(tiny):
    # per stage: one residual group plus one group inside each of the two blocks
    free = sorted(tiny.base_sizes[g] for g in tiny.graph.free_groups())
    assert free == sorted(w for w in BASE_WIDTHS for _ in range(3))
    # only the input (RGB) and the logits are fixed
    assert [tiny.base_sizes[g] for g in range(tiny.graph.num_groups) if tiny.graph.group_fixed[g]] == [3, 10]


@pytest.mark.parametrize("mult", MULTS)
def test_tiny_resnet_width_equals_standalone_model(tiny, mult):
    netaug = copy.deepcopy(tiny)
    netaug.set_width(mult)
    ref = _standalone_at(netaug, mult)

    # 1) size: parameter count and every layer's weight shape
    assert netaug.num_active_params() == sum(p.numel() for p in ref.parameters())
    for name, layer in netaug.dynamic_layers():
        ref_layer = ref.get_submodule(name)
        if isinstance(layer, (DynamicConv2d, DynamicLinear)):
            assert layer.active_weight().shape == ref_layer.weight.shape, name
        else:
            assert layer.active_in == ref_layer.num_features, name

    # 2) function: eval mode
    x = torch.randn(4, 3, 32, 32)
    netaug.eval(), ref.eval()
    torch.testing.assert_close(netaug(x), ref(x))

    # 3) function: train mode (batch statistics)
    netaug.train(), ref.train()
    torch.testing.assert_close(netaug(x), ref(x))


def test_tiny_resnet_base_updates_bn_like_standalone(tiny):
    netaug = copy.deepcopy(tiny)
    netaug.set_base()
    ref = _standalone_at(netaug, 1.0)
    netaug.train(), ref.train()
    x = torch.randn(4, 3, 32, 32)
    netaug(x), ref(x)
    for name, layer in netaug.dynamic_layers():
        if isinstance(layer, DynamicBatchNorm):
            n = layer.active_in
            torch.testing.assert_close(layer.running_mean[:n], ref.get_submodule(name).running_mean)
            torch.testing.assert_close(layer.running_var[:n], ref.get_submodule(name).running_var)


def test_tiny_resnet_parameter_counts():
    """The numbers reported by the experiment: x1 is ResNet-18 at 1/8 width."""
    netaug = NetAugModel(TinyResNet(BASE_WIDTHS), torch.randn(2, 3, 32, 32))
    counts = {}
    for m in MULTS:
        netaug.set_width(m)
        counts[m] = netaug.num_active_params()
    assert counts[1.0] == 176_402
    assert all(counts[a] < counts[b] for a, b in itertools.pairwise(MULTS))
    netaug.set_base()
    assert counts[1.0] == sum(p.numel() for p in netaug.export().parameters())


# ----------------------------------------------------------------------------------
# model level: any model from conftest, at random per-group sizes
# ----------------------------------------------------------------------------------
def _materialize(netaug: NetAugModel) -> nn.Module:
    """Replace every dynamic layer with a plain layer of the active size."""
    plain = copy.deepcopy(netaug.model)
    for name, layer in netaug.dynamic_layers():
        idx = torch.tensor(_index(layer.in_segments))
        n = len(idx)
        if isinstance(layer, DynamicConv2d):
            out = n if layer.depthwise else layer.active_out
            new: nn.Module = nn.Conv2d(
                n,
                out,
                cast(Any, layer.kernel_size),
                cast(Any, layer.stride),
                cast(Any, layer.padding),
                cast(Any, layer.dilation),
                groups=n if layer.depthwise else 1,
                bias=layer.bias is not None,
                padding_mode=layer.padding_mode,
            )
            w = layer.weight[idx] if layer.depthwise else layer.weight[:out][:, idx]
            b = None if layer.bias is None else (layer.bias[idx] if layer.depthwise else layer.bias[:out])
        elif isinstance(layer, DynamicLinear):
            new = nn.Linear(n, layer.active_out, bias=layer.bias is not None)
            w = layer.weight[: layer.active_out][:, idx]
            b = None if layer.bias is None else layer.bias[: layer.active_out]
        else:
            assert isinstance(layer, DynamicBatchNorm)
            new_bn = nn.BatchNorm2d(n, eps=layer.eps, momentum=layer.momentum, affine=layer.affine)
            w = None if layer.weight is None else layer.weight[idx]
            b = None if layer.bias is None else layer.bias[idx]
            assert (
                new_bn.running_mean is not None
                and new_bn.running_var is not None
                and layer.running_mean is not None
                and layer.running_var is not None
            )
            with torch.no_grad():
                new_bn.running_mean.copy_(layer.running_mean[idx])
                new_bn.running_var.copy_(layer.running_var[idx])
            new = new_bn
        with torch.no_grad():
            if w is not None and isinstance(new.weight, torch.Tensor):
                new.weight.copy_(w)
            if b is not None and isinstance(new.bias, torch.Tensor):
                new.bias.copy_(b)
        parent, _, child = name.rpartition(".")
        setattr(plain.get_submodule(parent) if parent else plain, child, new)
    return plain


@pytest.mark.parametrize("model_name", ["conv_chain", "tiny_dense", "resnet18", "mobilenet_v2", "efficientnet_b0"])
def test_random_sizes_equal_materialized_model(model_name):
    from conftest import build

    x = example_input()
    netaug = NetAugModel(build(model_name), x)
    gen = __import__("random").Random(0)
    for _ in range(3):
        netaug.set_random(gen)
        plain = _materialize(netaug)
        assert netaug.num_active_params() == sum(p.numel() for p in plain.parameters())
        netaug.eval(), plain.eval()
        torch.testing.assert_close(netaug(x), plain(x))


# ----------------------------------------------------------------------------------
# export at any width
# ----------------------------------------------------------------------------------
def _no_dynamic_layers(m: nn.Module) -> bool:
    return not any(type(x).__module__.startswith("torchnetaug") for x in m.modules())


@pytest.mark.parametrize("mult", MULTS)
def test_tiny_resnet_export_at_width_equals_standalone(tiny, mult):
    netaug = copy.deepcopy(tiny)
    netaug.set_width(mult)
    sizes = list(netaug.active_sizes)
    ref = _standalone_at(netaug, mult)
    netaug.set_base()  # export must not depend on the current mode
    out = netaug.export(sizes)
    assert _no_dynamic_layers(out)
    assert netaug.active_sizes == netaug.base_sizes and netaug.mode == "base"
    assert sum(p.numel() for p in out.parameters()) == sum(p.numel() for p in ref.parameters())
    assert {k: v.shape for k, v in out.state_dict().items()} == {k: v.shape for k, v in ref.state_dict().items()}
    x = torch.randn(4, 3, 32, 32)
    for train in (False, True):
        out.train(train), ref.train(train)
        torch.testing.assert_close(out(x), ref(x))
    # BN statistics of the exported model update like the standalone model
    out_any: Any = out
    ref_any: Any = ref
    torch.testing.assert_close(out_any.stem[1].running_mean, ref_any.stem[1].running_mean)


@pytest.mark.parametrize("model_name", ["conv_chain", "tiny_dense", "resnet18", "mobilenet_v2", "efficientnet_b0"])
def test_export_at_random_sizes_equals_dynamic_model(model_name):
    from conftest import build

    x = example_input()
    netaug = NetAugModel(build(model_name), x).eval()
    gen = __import__("random").Random(1)
    for _ in range(3):
        netaug.set_random(gen)
        out = netaug.export(netaug.active_sizes).eval()
        assert _no_dynamic_layers(out)
        assert sum(p.numel() for p in out.parameters()) == netaug.num_active_params()
        torch.testing.assert_close(out(x), netaug(x))
        torch.testing.assert_close(out(x), _materialize(netaug).eval()(x))
