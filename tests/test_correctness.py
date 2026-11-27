"""Correctness tests. Each one has a definite pass/fail answer (float64, tight tolerance)."""

import copy
import random
from typing import Any

import pytest
import torch
import torchvision
from conftest import assert_sorted, build, example_input, randomize
from torch import nn

from torchnetaug import DynamicConv2d, DynamicLinear, NetAugModel, netaug_step, recalibrate_bn

ATOL = 1e-9


def _out_fixed(aug: NetAugModel, layer: str) -> bool:
    g = aug.graph.layers[layer].out_group
    assert g is not None
    return aug.graph.group_fixed[g]


def wrap(name, mults=(1.0, 2.0, 3.0), **kw):
    base = build(name)
    x = example_input()
    aug = NetAugModel(base, x, width_mults=mults, **kw)
    return base, aug, x


@pytest.mark.parametrize("train_mode", [False, True])
def test_base_mode_equals_original(model_name, train_mode):
    """In base mode the wrapper must compute exactly what the original model computes."""
    base, aug, x = wrap(model_name)
    base.train(train_mode)
    aug.train(train_mode)
    torch.manual_seed(0)  # same dropout masks for both
    out_aug = aug(x)
    torch.manual_seed(0)
    torch.testing.assert_close(out_aug, base(x), atol=ATOL, rtol=0)


def test_has_augmentable_groups(model_name):
    _, aug, _ = wrap(model_name)
    assert aug.graph.free_groups(), aug.graph.summary()
    assert aug.max_sizes != aug.base_sizes


def test_export_equals_base(model_name):
    """export() returns the original class with the original shapes and the base function."""
    base, aug, x = wrap(model_name)
    randomize(aug, seed=7)  # train-like change: also touches the extra (augmented) channels
    aug.eval()
    exported = aug.export().eval()
    assert type(exported) is type(base)
    assert {k: v.shape for k, v in exported.state_dict().items()} == {k: v.shape for k, v in base.state_dict().items()}
    with aug.mode_ctx("base"):
        torch.testing.assert_close(exported(x), aug(x), atol=ATOL, rtol=0)


@pytest.mark.parametrize("train_mode", [False, True])
def test_sort_keeps_max_function(model_name, train_mode):
    """Sorting permutes channels consistently, so the max-width output must not change."""
    _, aug, x = wrap(model_name)
    randomize(aug, seed=3)
    aug.train(train_mode)
    for m in aug.modules():  # a dropout mask over channels is not permutation invariant
        if isinstance(m, nn.modules.dropout._DropoutNd):
            m.eval()
    aug.set_max()
    torch.manual_seed(0)  # same dropout masks before and after
    before = aug(x)
    perms = aug.sort_channels()
    assert perms, "nothing was sorted"
    assert any(not torch.equal(p, torch.arange(p.numel())) for p in perms.values())
    aug.set_max()
    torch.manual_seed(0)
    # train-mode batch norm over a 2-image batch amplifies float64 rounding (it depends on
    # the summation order, so on the thread count): about 2e-8 for efficientnet_b0
    tol = 1e-7 if train_mode else 1e-8
    torch.testing.assert_close(aug(x), before, atol=tol, rtol=tol)


def test_sort_orders_importance(model_name):
    _, aug, _ = wrap(model_name)
    randomize(aug, seed=4)
    aug.sort_channels()
    assert_sorted(aug)


def test_sort_only_outputs_breaks_function():
    """Negative control: permuting only the producers' output channels (and not the readers'
    inputs) changes the function. This is what happens with residual networks when a sort
    does not follow the channel groups."""
    _, aug, x = wrap("resnet18")
    randomize(aug, seed=5)
    aug.eval()
    aug.set_max()
    before = aug(x)
    params = dict(aug.model.named_parameters())
    g = max(aug.graph.free_groups(), key=lambda g: aug.max_sizes[g])
    perm = torch.randperm(aug.max_sizes[g], generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for name, dim, off, grp in aug._perm_ops()[g]:
            if dim == 0 and name in params:  # outputs only, readers left alone
                t = params[name]
                idx = torch.arange(off, off + perm.numel())
                t.index_copy_(dim, idx, t.index_select(dim, idx[perm]))
    assert not torch.allclose(aug(x), before, atol=1e-6)


@pytest.mark.parametrize("opt_name", ["sgd", "adam"])
def test_permutation_commutes_with_optimizer_step(opt_name):
    """step-then-permute must equal permute-then-step when optimizer state is permuted too."""
    _, aug, x = wrap("mobilenet_v2")
    randomize(aug, seed=6)
    aug.eval()  # no dropout, so both branches see the same function
    aug.set_max()
    target = torch.randint(0, 10, (x.shape[0],), generator=torch.Generator().manual_seed(0))

    def make_opt(m):
        if opt_name == "sgd":
            return torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        return torch.optim.Adam(m.parameters(), lr=0.01)

    def step(m, opt):
        opt.zero_grad()
        nn.functional.cross_entropy(m(x), target).backward()
        opt.step()

    opt = make_opt(aug)
    step(aug, opt)  # create optimizer state
    a, b = copy.deepcopy(aug), copy.deepcopy(aug)
    opt_a, opt_b = make_opt(a), make_opt(b)
    # load_state_dict may reuse the same state tensors, so give each optimizer its own copy
    opt_a.load_state_dict(copy.deepcopy(opt.state_dict()))
    opt_b.load_state_dict(copy.deepcopy(opt.state_dict()))

    g = torch.Generator().manual_seed(1)
    perms = {gid: torch.randperm(aug.max_sizes[gid], generator=g) for gid in aug.graph.free_groups()}

    a.apply_permutations(perms, opt_a)
    step(a, opt_a)
    step(b, opt_b)
    b.apply_permutations(perms, opt_b)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(pa, pb, atol=1e-10, rtol=1e-10, msg=n)


def test_sort_with_extra_model_keeps_ema_aligned():
    _, aug, _ = wrap("resnet18")
    randomize(aug, seed=8)
    ema = copy.deepcopy(aug)
    aug.sort_channels(extra_models=[ema])
    for (n, p), (_, q) in zip(aug.named_parameters(), ema.named_parameters()):
        torch.testing.assert_close(p, q, msg=n)


def test_random_mode_sizes_and_forward(model_name):
    _, aug, x = wrap(model_name)
    rng = random.Random(0)
    seen = set()
    for _ in range(8):
        aug.set_random(rng)
        for g, s in enumerate(aug.active_sizes):
            assert s in aug.choices[g]
        seen.add(tuple(aug.active_sizes))
        assert aug(x).shape == build(model_name)(x).shape
    assert len(seen) > 1
    aug.set_random(rng, granularity="global")
    aug(x)


def test_aug_pass_does_not_touch_bn_stats():
    _, aug, x = wrap("resnet18")
    aug.train()
    stats = {n: b.clone() for n, b in aug.named_buffers()}
    aug.set_max()
    aug(x)
    for n, b in aug.named_buffers():
        torch.testing.assert_close(b, stats[n], msg=n)
    aug.set_base()
    aug(x)
    assert any(not torch.equal(b, stats[n]) for n, b in aug.named_buffers() if "running_mean" in n)


def test_netaug_step_gradients():
    """Base-only training must leave the extra channels without gradient; the augmented
    pass must give them gradient."""
    _, aug, x = wrap("conv_chain")
    aug.train()
    target = torch.randint(0, 5, (2,))
    crit = nn.CrossEntropyLoss()

    netaug_step(aug, x, target, crit, augment=False)
    conv = aug.model.features[3]  # 8 -> 16 conv, max is 24 -> 48
    assert conv.weight.grad[:16, :8].abs().sum() > 0
    assert conv.weight.grad[16:].abs().sum() == 0
    assert conv.weight.grad[:, 8:].abs().sum() == 0

    aug.zero_grad()
    rng = random.Random(0)
    for _ in range(5):  # several samples so that some extra channels are surely used
        netaug_step(aug, x, target, crit, generator=rng)
    assert conv.weight.grad[16:].abs().sum() > 0
    assert aug.mode == "base"


def _pull_back(layer, attr, grad):
    """Place the gradient of a plain layer's tensor at the entries of the dynamic layer's
    full tensor that the current width uses (the adjoint of the forward slicing)."""
    p = getattr(layer, attr)
    with torch.enable_grad():
        if isinstance(layer, (DynamicConv2d, DynamicLinear)):
            sliced = layer.active_weight() if attr == "weight" else layer.active_bias()
        else:  # batch norm
            sliced = layer._select_in(p, 0)
        (full,) = torch.autograd.grad(sliced, p, grad_outputs=grad)
    return full


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_netaug_step_gradient_is_paper_eq2(model_name, seed):
    """One ``netaug_step`` accumulates exactly dL(base)/dW + alpha * dL(aug)/dW (paper Eq. 2).

    The reference uses two independent plain models: ``export()`` (base) and
    ``export(sizes)`` (the sampled augmented network). Each one's gradients are placed
    back into the shared max-width tensors and summed.
    """
    _, aug, x = wrap(model_name)
    randomize(aug, seed=3 + seed)  # extra channels get non-default values too
    aug.train()
    alpha = 0.7
    n_classes = aug(x).shape[1]
    target = torch.randint(0, n_classes, (x.shape[0],), generator=torch.Generator().manual_seed(seed))
    crit = nn.CrossEntropyLoss()

    aug.set_random(random.Random(seed))  # the same draw netaug_step makes below
    sizes = list(aug.active_sizes)
    aug.set_base()
    assert sizes != aug.base_sizes  # the test must exercise widening
    plain_base, plain_aug = aug.export().train(), aug.export(sizes).train()

    aug.zero_grad()
    torch.manual_seed(0)  # same dropout / stochastic-depth draws for both computations
    netaug_step(aug, x, target, crit, alpha=alpha, generator=random.Random(seed))
    torch.manual_seed(0)
    crit(plain_base(x), target).backward()
    (alpha * crit(plain_aug(x), target)).backward()

    expected = {n: torch.zeros_like(p) for n, p in aug.model.named_parameters()}
    for state, plain in ((aug.base_sizes, plain_base), (sizes, plain_aug)):
        aug.set_sizes(state)
        dynamic = dict(aug.dynamic_layers())
        for n, p in plain.named_parameters():
            if p.grad is None:
                continue
            owner, _, attr = n.rpartition(".")
            if owner in dynamic:
                expected[n] += _pull_back(dynamic[owner], attr, p.grad)
            else:
                expected[n] += p.grad
    aug.set_base()
    for n, p in aug.model.named_parameters():
        got = torch.zeros_like(p) if p.grad is None else p.grad
        torch.testing.assert_close(got, expected[n], atol=ATOL, rtol=1e-7, msg=n)


def test_recalibrate_bn_matches_exported_model():
    from torch.optim.swa_utils import update_bn

    _, aug, _ = wrap("mobilenet_v2")
    aug.set_max()  # recalibrate must switch to base by itself
    batches = [example_input(4, seed=s) for s in range(3)]
    recalibrate_bn(aug, batches)
    assert aug.mode == "max"
    exported = aug.export()
    reference = aug.export()
    update_bn(batches, reference)
    for (n, b), (_, r) in zip(exported.named_buffers(), reference.named_buffers()):
        if "running" in n:
            torch.testing.assert_close(b, r, atol=1e-10, rtol=1e-10, msg=n)


def test_fixed_groups_for_unknown_ops():
    _, aug, _ = wrap("group_norm")
    roles = aug.graph.layers
    assert "gn" not in roles  # GroupNorm is not whitelisted
    assert _out_fixed(aug, "c1")  # read by GroupNorm
    assert _out_fixed(aug, "c2")  # multiplied by a channel-wise parameter
    assert not _out_fixed(aug, "c3")


def test_resnet_stage_is_one_group():
    """Residual adds must merge every block output of a stage into one group."""
    _, aug, _ = wrap("resnet18")
    roles = aug.graph.layers
    stage1 = {roles["conv1"].out_group, roles["layer1.0.conv2"].out_group, roles["layer1.1.conv2"].out_group}
    assert len(stage1) == 1
    stage4 = {
        roles["layer4.0.conv2"].out_group,
        roles["layer4.0.downsample.0"].out_group,
        roles["layer4.1.conv2"].out_group,
    }
    assert len(stage4) == 1
    assert roles["fc"].in_groups == (roles["layer4.1.conv2"].out_group,)
    assert len(roles) == 1 + 16 + 3 + 1 + 20  # convs + fc + BNs


def test_mismatched_residual_layout_is_fixed():
    from conftest import ConcatMismatch

    base = randomize(ConcatMismatch())
    x = example_input()
    aug = NetAugModel(base, x)
    for name in ("a", "b", "proj"):
        assert _out_fixed(aug, name)
    assert not _out_fixed(aug, "c")
    torch.testing.assert_close(aug(x), base(x), atol=ATOL, rtol=0)


def test_concat_segments():
    _, aug, _ = wrap("tiny_dense")
    roles = aug.graph.layers
    s, c1, c2 = roles["stem"].out_group, roles["c1"].out_group, roles["c2"].out_group
    assert roles["c2"].in_groups == (s, c1)
    assert roles["bn"].in_groups == (s, c1, c2)


def test_set_width():
    _, aug, x = wrap("resnet18")
    aug.set_width(1.0)
    assert aug.mode == "base" and aug.active_sizes == aug.base_sizes
    aug.set_width(3.0)
    assert aug.active_sizes == aug.max_sizes
    aug.set_width(1.5)
    for g, (b, s) in enumerate(zip(aug.base_sizes, aug.active_sizes)):
        assert s == (b if aug.graph.group_fixed[g] else round(b * 1.5))
    aug(x)


def test_recalibrate_bn_at_other_width():
    """Calibrating at max width must give each BN the exact mean of its max-width inputs."""
    _, aug, _ = wrap("conv_chain")
    batches = [example_input(4, seed=s) for s in range(3)]
    bn = aug.model.features[1]
    seen = []
    hook = bn.register_forward_hook(lambda m, inp, out: seen.append(inp[0].detach()))
    aug.set_max()
    aug.train()
    with torch.no_grad():
        for b in batches:
            aug(b)
    hook.remove()
    expected = torch.stack([t.mean((0, 2, 3)) for t in seen]).mean(0)

    aug.set_base()
    recalibrate_bn(aug, batches, sizes=aug.max_sizes)
    assert aug.mode == "base"  # previous width restored
    torch.testing.assert_close(bn.running_mean, expected, atol=1e-10, rtol=1e-10)


class SpatialAttention(nn.Module):
    """x * sigmoid(conv(x)) with a 1-channel attention map (CBAM-style spatial attention)."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.att = nn.Conv2d(8, 1, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        y = torch.relu(self.c1(x))
        y = y * torch.sigmoid(self.att(y))
        return self.fc(self.c2(y).mean([2, 3]))


def test_one_channel_broadcast_group_is_fixed():
    x = example_input()
    m = NetAugModel(randomize(SpatialAttention()), x)
    assert _out_fixed(m, "att")
    assert m.graph.layers["c1"].out_group in m.graph.free_groups()
    m.set_max()
    m(x)
    rng = random.Random(0)
    for _ in range(5):
        m.set_random(rng)
        m(x)


def test_export_keeps_module_versions():
    # MnasNet's _load_from_state_dict reads the version stored in state_dict metadata
    x = example_input()
    original = randomize(torchvision.models.mnasnet0_5(num_classes=10)).eval()
    m = NetAugModel(original, x).eval()
    out = m.export().eval()
    torch.testing.assert_close(out(x), original(x))


def test_tied_weights_are_rejected():
    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(6, 6)
            self.b = nn.Linear(6, 6)
            self.b.weight = self.a.weight
            self.fc = nn.Linear(6, 2)

        def forward(self, x):
            return self.fc(torch.relu(self.b(torch.relu(self.a(x)))))

    with pytest.raises(ValueError, match="shared"):
        NetAugModel(Tied(), torch.randn(2, 6))


def test_binary_lower_rank_broadcast_is_fixed():
    """A lower-rank tracked tensor broadcast against spatial dims must stay fixed."""

    class LowerRank2D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.fc_bias = nn.Linear(3, 4)
            self.c2 = nn.Conv2d(8, 6, 3, padding=1)
            self.head = nn.Linear(6, 2)

        def forward(self, x):
            a = self.c1(x)
            b = self.fc_bias(x.mean((2, 3)))  # (1, 4) broadcasts onto spatial dim W=4
            return self.head(self.c2(a + b).mean((2, 3)))

    class LowerRank3D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.c_side = nn.Conv2d(3, 4, 3, padding=1)
            self.c2 = nn.Conv2d(8, 6, 3, padding=1)
            self.head = nn.Linear(6, 2)

        def forward(self, x):
            a = self.c1(x)
            b = self.c_side(x).mean(3)  # (1, 4, 4) broadcasts onto (H, W)=(4, 4)
            return self.head(self.c2(a + b).mean((2, 3)))

    x = example_input(batch=1, size=4)
    for cls, fixed_layer in ((LowerRank2D, "fc_bias"), (LowerRank3D, "c_side")):
        base = randomize(cls()).eval()
        aug = NetAugModel(base, x, width_mults=(1.0, 2.0)).eval()
        assert _out_fixed(aug, fixed_layer)
        assert not _out_fixed(aug, "c1")
        torch.testing.assert_close(aug(x), base(x), atol=ATOL, rtol=0)
        randomize(aug, seed=9)
        aug.set_max()
        before = aug(x)
        aug.sort_channels()
        torch.testing.assert_close(aug(x), before, atol=1e-8, rtol=1e-8)


def test_clamp_with_tensor_bounds_is_fixed():
    """torch.clamp with per-channel tensor/Parameter bounds must fix the touched groups."""

    class ClampParam(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.low = nn.Parameter(torch.randn(1, 8, 1, 1))
            self.c2 = nn.Conv2d(8, 6, 1)
            self.fc = nn.Linear(6, 2)

        def forward(self, x):
            y = torch.clamp(self.c1(x), min=self.low)
            return self.fc(self.c2(y).mean((2, 3)))

    class ClampTracked(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.c2 = nn.Conv2d(3, 8, 3, padding=1)
            self.c3 = nn.Conv2d(8, 6, 1)
            self.fc = nn.Linear(6, 2)

        def forward(self, x):
            y = torch.clamp(self.c1(x), min=self.c2(x))
            return self.fc(self.c3(y).mean((2, 3)))

    class ClampScalar(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.c2 = nn.Conv2d(8, 6, 1)
            self.fc = nn.Linear(6, 2)

        def forward(self, x):
            y = torch.clamp(self.c1(x), min=0.0, max=6.0)
            return self.fc(self.c2(y).mean((2, 3)))

    x = example_input(batch=2, size=8)
    for cls, fixed_names in ((ClampParam, ("c1",)), (ClampTracked, ("c1", "c2"))):
        base = randomize(cls()).eval()
        aug = NetAugModel(base, x, width_mults=(1.0, 2.0)).eval()
        for name in fixed_names:
            assert _out_fixed(aug, name)
        torch.testing.assert_close(aug(x), base(x), atol=ATOL, rtol=0)
        randomize(aug, seed=9)
        aug.set_max()
        before = aug(x)
        aug.sort_channels()
        torch.testing.assert_close(aug(x), before, atol=1e-8, rtol=1e-8)

    aug_scalar = NetAugModel(randomize(ClampScalar()).eval(), x, width_mults=(1.0, 2.0)).eval()
    assert not _out_fixed(aug_scalar, "c1")


def test_reshape_hardcoded_and_size1_squeeze_are_fixed():
    """Hardcoded channel sizes in view/reshape and squeezing/flattening dim 1 must fix C."""

    class HardcodedView(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(self.c1(x).mean((2, 3)).view(x.size(0), 8))

    class HardcodedMixView(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            y = self.c1(x).view(x.size(0), 8, -1)
            return self.fc(y.mean(2))

    class HardcodedBatchView(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(self.c1(x).mean((2, 3)).view(-1, 8))

    class SqueezeCh1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 1, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 4))
            self.fc = nn.Linear(1, 4)

        def forward(self, x):
            y = self.pool(self.c1(x)).squeeze(1)
            return self.fc(y.mean(2))

    class Flatten01Ch1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 1, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 4))
            self.fc = nn.Linear(1, 4)

        def forward(self, x):
            y = torch.flatten(self.pool(self.c1(x)), 0, 1)
            return self.fc(y.mean(2))

    class DynamicViews(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.c2 = nn.Conv2d(8, 6, 3, padding=1)
            self.fc = nn.Linear(6, 4)

        def forward(self, x):
            y = torch.relu(self.c1(x))
            b, c, h, w = y.shape
            y = y.reshape(b, c, h * w).view(b, -1, h, w)
            z = self.c2(y).mean((2, 3), keepdim=True).view(b, -1)
            return self.fc(z)

    x2 = example_input(batch=2, size=8)
    for cls, inp in (
        (HardcodedView, x2),
        (HardcodedMixView, x2),
        (HardcodedBatchView, x2),
        (SqueezeCh1, x2),
        (Flatten01Ch1, example_input(batch=1, size=8)),
    ):
        base = randomize(cls()).eval()
        aug = NetAugModel(base, inp, width_mults=(1.0, 2.0)).eval()
        assert _out_fixed(aug, "c1"), cls.__name__
        aug.set_max()
        torch.testing.assert_close(aug(inp), base(inp), atol=ATOL, rtol=0)

    dyn_base = randomize(DynamicViews()).eval()
    dyn_aug = NetAugModel(dyn_base, x2, width_mults=(1.0, 2.0)).eval()
    assert not _out_fixed(dyn_aug, "c1")
    assert not _out_fixed(dyn_aug, "c2")
    randomize(dyn_aug, seed=11)
    dyn_aug.set_max()
    before = dyn_aug(x2)
    dyn_aug.sort_channels()
    torch.testing.assert_close(dyn_aug(x2), before, atol=1e-8, rtol=1e-8)


def test_batchnorm_no_affine_no_stats():
    """BatchNorm(affine=False, track_running_stats=False) has no params or buffers."""

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8, affine=False, track_running_stats=False)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(torch.relu(self.bn(self.c1(x))).mean((2, 3)))

    x = example_input(batch=4, size=8)
    base = randomize(Net())
    aug = NetAugModel(base, x, width_mults=(1.0, 2.0))
    assert not _out_fixed(aug, "c1")
    for train_mode in (False, True):
        base.train(train_mode), aug.train(train_mode)
        torch.testing.assert_close(aug(x), base(x), atol=ATOL, rtol=0)
    randomize(aug, seed=5)
    aug.eval()
    aug.set_max()
    before = aug(x)
    aug.sort_channels()
    torch.testing.assert_close(aug(x), before, atol=1e-8, rtol=1e-8)
    exp_max = aug.export(aug.max_sizes).eval()
    torch.testing.assert_close(exp_max(x), aug(x), atol=ATOL, rtol=0)
    recalibrate_bn(aug, [x])
    aug.set_base()
    torch.testing.assert_close(aug.export().eval()(x), aug(x), atol=ATOL, rtol=0)


def test_sync_batchnorm_is_fixed():
    """SyncBatchNorm must not be silently replaced by local DynamicBatchNorm."""

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.SyncBatchNorm(8)
            self.c2 = nn.Conv2d(8, 6, 1)
            self.fc = nn.Linear(6, 4)

        def forward(self, x):
            return self.fc(self.c2(torch.relu(self.bn(self.c1(x)))).mean((2, 3)))

    x = example_input(batch=2, size=8)
    base = randomize(Net()).eval()
    aug = NetAugModel(base, x, width_mults=(1.0, 2.0)).eval()
    assert "bn" not in aug.graph.layers
    assert isinstance(aug.model.bn, nn.SyncBatchNorm)
    assert _out_fixed(aug, "c1")
    assert not _out_fixed(aug, "c2")
    torch.testing.assert_close(aug(x), base(x), atol=ATOL, rtol=0)
    aug.set_max()
    aug(x)


def test_export_sizes_keeps_float64_precision_under_float32_default():
    """export(sizes) must not truncate float64 weights through the default float32 dtype."""

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(torch.relu(self.bn(self.c1(x))).mean((2, 3)))

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        base = randomize(Net().double(), seed=2).eval()
        x = torch.randn(2, 3, 8, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        aug = NetAugModel(base, x, width_mults=(1.0, 2.0)).eval()
        randomize(aug, seed=3)
        aug.set_max()
        exported: Any = aug.export(aug.active_sizes).eval()
        c1: Any = aug.model.c1
        torch.testing.assert_close(exported.c1.weight, c1.active_weight(), rtol=0, atol=0)
        torch.testing.assert_close(exported(x), aug(x), rtol=0, atol=0)
    finally:
        torch.set_default_dtype(previous)


def test_export_preserves_per_module_training_and_requires_grad():
    """Both export() and export(sizes) must preserve frozen BN flags and requires_grad."""

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(8)
            self.c2 = nn.Conv2d(8, 6, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(6)
            self.fc = nn.Linear(6, 4)

        def forward(self, x):
            x = torch.relu(self.bn1(self.c1(x)))
            return self.fc(torch.relu(self.bn2(self.c2(x))).mean((2, 3)))

    x = example_input(batch=2, size=8)
    base: Any = randomize(Net()).train()
    base.bn1.eval()
    base.c1.weight.requires_grad_(False)
    aug = NetAugModel(base, x, width_mults=(1.0, 2.0))
    # also freeze a parameter after wrapping
    c2: Any = aug.model.c2
    c2.bias.requires_grad_(False)

    for exported in (aug.export(), aug.export(aug.max_sizes)):
        exp: Any = exported
        assert not exp.bn1.training
        assert exp.bn2.training
        assert not exp.c1.weight.requires_grad
        assert exp.c1.bias.requires_grad
        assert not exp.c2.bias.requires_grad
        assert exp.c2.weight.requires_grad
