"""The lemmas that the correctness proof in docs/design.md section 5.1 rests on.

The proof needs one property from every operation that lets channels pass through:
it must be *channel-separable*, i.e. output channel c depends only on input channel c
(with the same function for every channel). Two consequences are checked directly:

* permutation:  f(x[:, P]) == f(x)[:, P]   for a random permutation P (sorting);
* selection:    f(x[:, S]) == f(x)[:, S]   for a random channel subset S (narrower widths).

The cases are generated from the whitelists in ``torchnetaug.graph``, so adding an
operation to a whitelist without a case here fails ``test_every_whitelisted_op_has_a_case``.

The dynamic layers get the matching lemma for their weights:
layer(x[:, P_in]; W[P_out][:, P_in]) == layer(x; W)[:, P_out].
"""

from __future__ import annotations

import operator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchnetaug import graph as G
from torchnetaug.layers import DynamicBatchNorm, DynamicConv2d, DynamicLinear

B, C, H, W = 3, 7, 6, 6


def _x(shape=(B, C, H, W)):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(1))


# One callable x -> y per whitelisted entry. Dropout runs in eval mode: in training it
# draws an independent mask per element, so it is separable only in distribution.
MODULE_CASES = {
    nn.ReLU: nn.ReLU(), nn.ReLU6: nn.ReLU6(), nn.SiLU: nn.SiLU(), nn.GELU: nn.GELU(),
    nn.Hardswish: nn.Hardswish(), nn.Hardsigmoid: nn.Hardsigmoid(), nn.Sigmoid: nn.Sigmoid(),
    nn.Tanh: nn.Tanh(), nn.LeakyReLU: nn.LeakyReLU(0.1), nn.ELU: nn.ELU(), nn.Mish: nn.Mish(),
    nn.Identity: nn.Identity(), nn.Dropout: nn.Dropout(0.5).eval(),
    nn.Dropout1d: nn.Dropout1d(0.5).eval(), nn.Dropout2d: nn.Dropout2d(0.5).eval(),
    nn.Dropout3d: nn.Dropout3d(0.5).eval(), nn.AlphaDropout: nn.AlphaDropout(0.5).eval(),
    nn.MaxPool1d: nn.MaxPool1d(2), nn.MaxPool2d: nn.MaxPool2d(3, 2, 1),
    nn.AvgPool1d: nn.AvgPool1d(2), nn.AvgPool2d: nn.AvgPool2d(2),
    nn.AdaptiveAvgPool1d: nn.AdaptiveAvgPool1d(2), nn.AdaptiveAvgPool2d: nn.AdaptiveAvgPool2d(1),
    nn.AdaptiveMaxPool1d: nn.AdaptiveMaxPool1d(2), nn.AdaptiveMaxPool2d: nn.AdaptiveMaxPool2d(2),
    nn.Upsample: nn.Upsample(scale_factor=2, mode="bilinear"),
    nn.UpsamplingNearest2d: nn.UpsamplingNearest2d(scale_factor=2),
    nn.UpsamplingBilinear2d: nn.UpsamplingBilinear2d(scale_factor=2),
}
ONE_D = {nn.MaxPool1d, nn.AvgPool1d, nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d, nn.Dropout1d}

FUNCTION_CASES = {
    F.relu: F.relu, F.relu_: lambda x: F.relu_(x.clone()), F.relu6: F.relu6, F.silu: F.silu,
    F.gelu: F.gelu, F.hardswish: F.hardswish, F.hardsigmoid: F.hardsigmoid, F.sigmoid: F.sigmoid,
    F.tanh: F.tanh, F.leaky_relu: F.leaky_relu, F.elu: F.elu, F.mish: F.mish,
    F.dropout: lambda x: F.dropout(x, 0.5, training=False),
    F.dropout1d: lambda x: F.dropout1d(x.flatten(2), 0.5, training=False),
    F.dropout2d: lambda x: F.dropout2d(x, 0.5, training=False),
    F.dropout3d: lambda x: F.dropout3d(x.unsqueeze(2), 0.5, training=False),
    F.max_pool1d: lambda x: F.max_pool1d(x.flatten(2), 2),
    F.max_pool2d: lambda x: F.max_pool2d(x, 2),
    F.avg_pool1d: lambda x: F.avg_pool1d(x.flatten(2), 2),
    F.avg_pool2d: lambda x: F.avg_pool2d(x, 2),
    F.adaptive_avg_pool1d: lambda x: F.adaptive_avg_pool1d(x.flatten(2), 3),
    F.adaptive_avg_pool2d: lambda x: F.adaptive_avg_pool2d(x, 1),
    F.adaptive_max_pool1d: lambda x: F.adaptive_max_pool1d(x.flatten(2), 3),
    F.adaptive_max_pool2d: lambda x: F.adaptive_max_pool2d(x, 2),
    F.interpolate: lambda x: F.interpolate(x, scale_factor=2, mode="bicubic"),
    torch.relu: torch.relu, torch.relu_: lambda x: torch.relu_(x.clone()),
    torch.sigmoid: torch.sigmoid, torch.tanh: torch.tanh,
    torch.clamp: lambda x: torch.clamp(x, -0.5, 0.5), torch.clip: lambda x: torch.clip(x, -0.5, 0.5),
}
try:
    from torchvision.ops import stochastic_depth

    # "row" mode draws one value per sample, so with the same seed the mask is identical
    # whatever the channel count or order; training=True is checked under a fixed seed.
    FUNCTION_CASES[stochastic_depth] = lambda x: stochastic_depth(x, 0.5, "row", training=True)
except ImportError:  # pragma: no cover
    pass

METHOD_CASES = {name: (lambda name: lambda x: getattr(x.clone(), name)())(name)
                for name in ["relu", "relu_", "sigmoid", "sigmoid_", "tanh", "tanh_",
                             "contiguous", "clone", "detach"]}
METHOD_CASES.update({name: (lambda name: lambda x: getattr(x.clone(), name)(-0.5, 0.5))(name)
                     for name in ["clamp", "clamp_", "clip", "clip_"]})

# Reshapes and reductions pass channels only when dims 0 and 1 are unchanged; these are
# the shapes the analyser accepts (e.g. flatten after global pooling).
RESHAPE_CASES = {
    "flatten": lambda x: x.mean([2, 3], keepdim=True).flatten(1),
    "torch.flatten": lambda x: torch.flatten(x.mean([2, 3], keepdim=True), 1),
    "view": lambda x: x.mean([2, 3], keepdim=True).view(x.shape[0], x.shape[1]),
    "reshape": lambda x: x.reshape(x.shape[0], x.shape[1], -1),
    "torch.reshape": lambda x: torch.reshape(x, (x.shape[0], x.shape[1], -1)),
    "squeeze": lambda x: x.mean([2, 3], keepdim=True).squeeze(-1),
    "torch.squeeze": lambda x: torch.squeeze(x.mean(3, keepdim=True), 3),
    "mean": lambda x: x.mean([2, 3]), "torch.mean": lambda x: torch.mean(x, (2, 3), keepdim=True),
    "sum": lambda x: x.sum(2), "torch.sum": lambda x: torch.sum(x, [2, 3]),
    "amax": lambda x: x.amax(3), "torch.amax": lambda x: torch.amax(x, (2, 3)),
}

BINARY_FUNCS = sorted(G._BINARY_FUNCTIONS, key=repr) + sorted(G._BINARY_METHODS)


def _binary(op):
    if isinstance(op, str):
        return lambda a, b: getattr(a.clone(), op)(b)
    if op in (operator.iadd, operator.isub, operator.imul, operator.itruediv):
        return lambda a, b: op(a.clone(), b)
    return op


def _check_separable(f, x):
    torch.manual_seed(0)
    y = f(x)
    perm = torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(2))
    sub = torch.tensor([0, 2, 3, 6])[: x.shape[1]]
    torch.manual_seed(0)
    torch.testing.assert_close(f(x[:, perm]), y[:, perm], rtol=1e-6, atol=1e-6)
    torch.manual_seed(0)
    torch.testing.assert_close(f(x[:, sub]), y[:, sub], rtol=1e-6, atol=1e-6)


def test_every_whitelisted_op_has_a_case():
    assert set(G._PASSTHROUGH_MODULES) <= set(MODULE_CASES)
    assert G._PASSTHROUGH_FUNCTIONS <= set(FUNCTION_CASES)
    assert G._PASSTHROUGH_METHODS <= set(METHOD_CASES)
    assert {"flatten", "view", "reshape", "squeeze"} == G._RESHAPE_METHODS
    assert {torch.flatten, torch.reshape, torch.squeeze} == G._RESHAPE_FUNCTIONS
    assert G._MEAN_FUNCTIONS == {torch.mean, torch.amax, torch.sum}
    assert G._MEAN_METHODS == {"mean", "amax", "sum"}


@pytest.mark.parametrize("cls", list(MODULE_CASES), ids=lambda c: c.__name__)
def test_module_is_channel_separable(cls):
    x = _x((B, C, 8)) if cls in ONE_D else _x()
    _check_separable(MODULE_CASES[cls], x)


@pytest.mark.parametrize("fn", list(FUNCTION_CASES), ids=lambda f: getattr(f, "__name__", repr(f)))
def test_function_is_channel_separable(fn):
    _check_separable(FUNCTION_CASES[fn], _x())


@pytest.mark.parametrize("name", list(METHOD_CASES))
def test_method_is_channel_separable(name):
    _check_separable(METHOD_CASES[name], _x())


@pytest.mark.parametrize("name", list(RESHAPE_CASES))
def test_reshape_and_reduce_are_channel_separable(name):
    _check_separable(RESHAPE_CASES[name], _x())


@pytest.mark.parametrize("op", BINARY_FUNCS, ids=repr)
@pytest.mark.parametrize("other_shape", [(B, C, H, W), (B, C, 1, 1), (B, 1, H, W), ()], ids=str)
def test_binary_op_with_the_same_permutation_is_separable(op, other_shape):
    # Both sides of a residual add / SE multiply belong to one group, so they get the same
    # permutation. A 1-channel or scalar side broadcasts and is not permuted at all.
    f = _binary(op)
    a = _x()
    b = (torch.rand(other_shape, generator=torch.Generator().manual_seed(3)) + 0.5) if other_shape else 0.7
    per_channel = isinstance(b, torch.Tensor) and b.dim() > 1 and b.shape[1] == C
    y = f(a, b)
    perm = torch.randperm(C, generator=torch.Generator().manual_seed(2))
    sub = torch.tensor([0, 2, 3, 6])
    for idx in (perm, sub):
        torch.testing.assert_close(f(a[:, idx], b[:, idx] if per_channel else b), y[:, idx])


def test_cat_is_block_permutation():
    # cat along channels: each input keeps its own permutation inside its own slot
    a, b = _x((B, 3, H, W)), _x((B, 4, H, W)) * 2
    pa, pb = torch.tensor([2, 0, 1]), torch.tensor([3, 1, 0, 2])
    y = torch.cat([a, b], 1)
    block = torch.cat([pa, 3 + pb])
    torch.testing.assert_close(torch.cat([a[:, pa], b[:, pb]], 1), y[:, block])


# ----------------------------------------------------------------------------------
# dynamic layers: permuting weights on both sides commutes with the layer
# ----------------------------------------------------------------------------------
def test_dense_conv_lemma():
    conv = DynamicConv2d(5, 6, 3, padding=1)
    x = _x((B, 5, H, W))
    p_in, p_out = torch.randperm(5), torch.randperm(6)
    y = conv(x)
    with torch.no_grad():
        conv.weight.copy_(conv.weight[p_out][:, p_in])
        conv.bias.copy_(conv.bias[p_out])
    torch.testing.assert_close(conv(x[:, p_in]), y[:, p_out])


def test_depthwise_conv_lemma():
    conv = DynamicConv2d(6, 6, 3, padding=1, groups=6)
    x = _x((B, 6, H, W))
    p = torch.randperm(6)
    y = conv(x)
    with torch.no_grad():
        conv.weight.copy_(conv.weight[p])
        conv.bias.copy_(conv.bias[p])
    torch.testing.assert_close(conv(x[:, p]), y[:, p])


def test_linear_lemma():
    lin = DynamicLinear(5, 4)
    x = _x((B, 5))
    p_in, p_out = torch.randperm(5), torch.randperm(4)
    y = lin(x)
    with torch.no_grad():
        lin.weight.copy_(lin.weight[p_out][:, p_in])
        lin.bias.copy_(lin.bias[p_out])
    torch.testing.assert_close(lin(x[:, p_in]), y[:, p_out])


@pytest.mark.parametrize("training", [False, True])
def test_batchnorm_lemma(training):
    bn = DynamicBatchNorm(6)
    with torch.no_grad():
        for t in (bn.weight, bn.bias, bn.running_mean):
            t.normal_()
        bn.running_var.uniform_(0.5, 2)
    x = _x((B, 6, H, W))
    p = torch.randperm(6)
    bn.train(training)
    ref = bn.__class__(6)
    ref.load_state_dict(bn.state_dict())
    ref.train(training)
    y = ref(x)
    with torch.no_grad():
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(bn, name).copy_(getattr(bn, name)[p])
    torch.testing.assert_close(bn(x[:, p]), y[:, p])
    torch.testing.assert_close(bn.running_mean, ref.running_mean[p])
    torch.testing.assert_close(bn.running_var, ref.running_var[p])
