"""Using the dynamic layers directly, and the guards around them.

* A network built by hand from dynamic layers, with widths set by hand, computes exactly
  what a plain network loaded with the sliced weights computes.
* ``NetAugModel`` and ``analyze`` expect the *tiny* model made of plain layers. A model
  that already contains dynamic layers is rejected with a clear error instead of being
  silently widened again (which would also make ``export()`` depend on this package).
* ``export(sizes)`` keeps the model's dtype even when the default dtype differs.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from conftest import randomize

from torchnetaug import (
    DynamicBatchNorm,
    DynamicConv2d,
    DynamicLinear,
    DynamicModule,
    NetAugModel,
    analyze,
)


def _net(conv, bn, lin, w1, w2):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1, self.b1 = conv(3, w1, 3, padding=1, bias=False), bn(w1)
            self.c2, self.b2 = conv(w1, w2, 3, padding=1, bias=False), bn(w2)
            self.fc = lin(w2, 10)

        def forward(self, x):
            x = torch.relu(self.b1(self.c1(x)))
            x = torch.relu(self.b2(self.c2(x)))
            return self.fc(x.mean((2, 3)))

    return Net()


@pytest.mark.parametrize("train", [False, True])
def test_hand_built_dynamic_network_equals_plain_network(train):
    torch.manual_seed(0)
    dyn = randomize(_net(DynamicConv2d, DynamicBatchNorm, DynamicLinear, 24, 48))
    # width by hand: producers get active_out, readers get the matching input segments
    dyn.c1.active_out = 8
    dyn.b1.set_in_segments([(0, 8)])
    dyn.c2.set_in_segments([(0, 8)])
    dyn.c2.active_out = 16
    dyn.b2.set_in_segments([(0, 16)])
    dyn.fc.set_in_segments([(0, 16)])

    plain = _net(nn.Conv2d, nn.BatchNorm2d, nn.Linear, 8, 16)
    state = {}
    for name, t in dyn.state_dict().items():
        shape = plain.state_dict()[name].shape
        state[name] = t[tuple(slice(0, s) for s in shape)].clone()
    plain.load_state_dict(state)

    dyn.train(train), plain.train(train)
    x = torch.randn(4, 3, 16, 16)
    torch.testing.assert_close(dyn(x), plain(x), rtol=0, atol=0)
    if train:  # the active slice of the running statistics was updated like the plain BN
        torch.testing.assert_close(dyn.b2.running_mean[:16], plain.b2.running_mean, rtol=0, atol=0)
        torch.testing.assert_close(dyn.b2.running_var[:16], plain.b2.running_var, rtol=0, atol=0)


def test_analyze_rejects_dynamic_layers():
    model = _net(DynamicConv2d, DynamicBatchNorm, DynamicLinear, 8, 16)
    with pytest.raises(TypeError, match="c1.*DynamicConv2d"):
        analyze(model, torch.randn(2, 3, 16, 16))


def test_netaug_rejects_dynamic_layers_with_tracing():
    model = _net(DynamicConv2d, DynamicBatchNorm, DynamicLinear, 8, 16)
    with pytest.raises(TypeError, match="plain"):
        NetAugModel(model, torch.randn(2, 3, 16, 16))


def test_netaug_rejects_dynamic_layers_with_hand_built_graph():
    x = torch.randn(2, 3, 16, 16)
    graph = analyze(_net(nn.Conv2d, nn.BatchNorm2d, nn.Linear, 8, 16), x)
    model = _net(DynamicConv2d, DynamicBatchNorm, DynamicLinear, 8, 16)
    with pytest.raises(TypeError, match="plain"):
        NetAugModel(model, graph=graph)


def test_rejects_a_netaug_model():
    x = torch.randn(2, 3, 16, 16)
    inner = NetAugModel(_net(nn.Conv2d, nn.BatchNorm2d, nn.Linear, 8, 16), x)
    with pytest.raises(TypeError):
        NetAugModel(inner.model, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_export_sizes_keeps_model_dtype(dtype):
    assert torch.get_default_dtype() == torch.float64  # set in conftest: differs from dtype
    torch.manual_seed(0)
    model = _net(nn.Conv2d, nn.BatchNorm2d, nn.Linear, 8, 16).to(dtype)
    x = torch.randn(2, 3, 16, 16, dtype=dtype)
    netaug = NetAugModel(model, x, width_mults=(1.0, 2.0)).eval()
    netaug.set_width(2.0)
    exported = netaug.export(netaug.active_sizes).eval()
    assert not any(isinstance(m, DynamicModule) for m in exported.modules())
    assert {t.dtype for t in exported.state_dict().values() if t.is_floating_point()} == {dtype}
    with torch.no_grad():
        torch.testing.assert_close(exported(x), netaug(x), rtol=0, atol=0)
