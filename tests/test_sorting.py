"""Channel sorting: importance formula, permutation invariance, optimizer equivariance and
input checks of ``apply_permutations`` / ``sort_channels`` (float64, tight tolerances)."""

import copy
import math
from collections.abc import Callable
from typing import cast

import pytest
import torch
from conftest import build, example_input, randomize
from test_random_architectures import N_SEEDS, RandomNet
from torch import nn

from torchnetaug import ChannelGraph, LayerRole, NetAugModel

REL = 1e-9  # T-inv tolerance (relative, element by element)
OPT_REL = 1e-12  # T-opt tolerance (max|a-b| / max|b| per tensor)


def wrap(name: str, mults=(1.0, 2.0, 3.0), **kw) -> NetAugModel:
    return NetAugModel(build(name), example_input(), width_mults=mults, **kw)


def random_aug(seed: int) -> NetAugModel:
    torch.manual_seed(seed)
    x = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(seed))
    aug = NetAugModel(randomize(RandomNet(seed), seed=seed), x, width_mults=(1.0, 1.5, 2.5))
    randomize(aug, seed=seed + 2000)  # random values in the extra channels too
    return aug


def is_identity(p: torch.Tensor) -> bool:
    return torch.equal(p, torch.arange(p.numel()))


def assert_elementwise_rel(new: torch.Tensor, old: torch.Tensor, msg: str) -> None:
    """``|new - old| <= REL * |old|`` for every element."""
    assert new.shape == old.shape, msg
    bad = (new - old).abs() > REL * old.abs()
    assert not bad.any(), f"{msg}: max rel deviation {((new - old).abs() / old.abs()).max().item():.3g}"


def rel_dev(a: torch.Tensor, b: torch.Tensor) -> float:
    """``max|a - b| / max|b|``; 0 when both are all zero, inf when only ``b`` is."""
    a, b = a.detach().double(), b.detach().double()
    assert a.shape == b.shape
    if a.numel() == 0:
        return 0.0
    diff, scale = (a - b).abs().max().item(), b.abs().max().item()
    if scale == 0:
        return 0.0 if diff == 0 else math.inf
    return diff / scale


def snapshot(aug: NetAugModel, opt: torch.optim.Optimizer | None = None) -> tuple[dict, list]:
    state = {k: v.detach().clone() for k, v in aug.state_dict().items()}
    opt_state = []
    if opt is not None:
        for p in aug.parameters():
            opt_state.append({k: v.clone() if torch.is_tensor(v) else v for k, v in opt.state.get(p, {}).items()})
    return state, opt_state


def assert_unchanged(aug: NetAugModel, opt: torch.optim.Optimizer | None, snap: tuple[dict, list]) -> None:
    state, opt_state = snapshot(aug, opt)
    assert state.keys() == snap[0].keys()
    for k in state:
        assert torch.equal(state[k], snap[0][k]), f"{k} changed"
    assert len(opt_state) == len(snap[1])
    for i, (s, t) in enumerate(zip(opt_state, snap[1])):
        assert s.keys() == t.keys()
        for k in s:
            assert torch.equal(s[k], t[k]) if torch.is_tensor(s[k]) else s[k] == t[k], f"optimizer state {i}.{k}"


# --------------------------------------------------------------------------------------
# T-inv: importance of a group ignores the other groups' order, follows its own order
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(N_SEEDS))
def test_importance_invariant_to_other_groups(seed):
    aug = random_aug(seed)
    old = aug.channel_importance()
    free = aug.graph.free_groups()
    assert free and set(old) == set(free)
    gen = torch.Generator().manual_seed(seed)
    own_moved = 0
    for g in free:
        others = {h: torch.randperm(aug.max_sizes[h], generator=gen) for h in free if h != g}
        moved = copy.deepcopy(aug)
        moved.apply_permutations(others)
        assert_elementwise_rel(moved.channel_importance()[g], old[g], f"group {g} after permuting the others")

        p = torch.randperm(aug.max_sizes[g], generator=gen)
        own_moved += not is_identity(p)
        own = copy.deepcopy(aug)
        own.apply_permutations({g: p})
        assert_elementwise_rel(own.channel_importance()[g], old[g][p], f"group {g} after permuting itself")
    assert own_moved > 0, "every own-group permutation was the identity"


def test_importance_invariance_is_not_vacuous():
    """Most seeds have two or more free groups, so 'permute the others' really moves channels."""
    multi = 0
    for seed in range(N_SEEDS):
        aug = random_aug(seed)
        free = aug.graph.free_groups()
        multi += len(free) >= 2 and any(aug.max_sizes[h] > 1 for h in free)
    assert multi > N_SEEDS // 2, multi


# --------------------------------------------------------------------------------------
# T-official: MobileNetV2 expand groups use the official NetAug formula
# --------------------------------------------------------------------------------------
def test_importance_matches_official_mobilenet_v2():
    """Official NetAug sorts an inverted block's expand channels by the L1 norm of the 1x1
    project conv over its input channels: ``|W_project|.sum(dim=(0, 2, 3))``."""
    aug = wrap("mobilenet_v2")
    randomize(aug, seed=11)
    importance = aug.channel_importance()
    checked = 0
    for k, block in enumerate(cast(nn.Sequential, aug.model.get_submodule("features"))):
        producer = f"features.{k}.conv.0.0"
        role = aug.graph.layers.get(producer)
        if role is None or role.kind != "dense":
            continue  # not an inverted block, or expand ratio 1 (conv.0.0 is the depthwise conv)
        g = role.out_group
        assert g is not None and not aug.graph.group_fixed[g], producer
        project = block.get_submodule("conv.2")
        assert isinstance(project, nn.Conv2d)
        expected = project.weight.detach().abs().sum(dim=(0, 2, 3))
        torch.testing.assert_close(importance[g], expected, rtol=1e-12, atol=0, msg=producer)
        checked += 1
    assert checked >= 10, checked


# --------------------------------------------------------------------------------------
# Fix 1: groups without a dense reader use only the dense producers' weights
# --------------------------------------------------------------------------------------
class ProducerOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.dw = nn.Conv2d(4, 4, 3, padding=1, groups=4)
        self.lonely_bn = nn.BatchNorm2d(5)

    def forward(self, x):
        return self.dw(self.bn(self.conv(x)))


def test_fallback_importance_uses_only_dense_producers():
    """Hand-written graph: group 1 has a producer, a BN and a depthwise member but no dense
    reader; group 2 has only a BN member (no producer, no reader)."""
    graph = ChannelGraph(
        group_sizes=[3, 4, 5],
        group_fixed=[True, False, False],
        layers={
            "conv": LayerRole("dense", (0,), 1),
            "bn": LayerRole("member", (1,)),
            "dw": LayerRole("member", (1,)),
            "lonely_bn": LayerRole("member", (2,)),
        },
    )
    aug = NetAugModel(ProducerOnly(), graph=graph, width_mults=(1.0, 2.0))
    randomize(aug, seed=2)
    imp = aug.channel_importance()
    expected = cast(nn.Conv2d, aug.model.get_submodule("conv")).weight.detach().abs().sum(dim=(1, 2, 3))
    torch.testing.assert_close(imp[1], expected, rtol=1e-12, atol=0)
    assert torch.equal(imp[2], torch.zeros(10))


def test_fallback_importance_on_analyzed_graphs():
    """``analyze()`` produces groups without a dense reader (dead branches), some with BN or
    depthwise members. Their importance is the sum of the producers' output-weight L1 norms."""
    groups = with_members = 0
    for seed in range(N_SEEDS):
        aug = random_aug(seed)
        imp = aug.channel_importance()
        layers = aug.graph.layers
        for g in aug.graph.free_groups():
            if any(r.kind == "dense" and g in r.in_groups for r in layers.values()):
                continue
            producers = [n for n, r in layers.items() if r.kind == "dense" and r.out_group == g]
            assert producers, (seed, g)
            weights = [cast(nn.Conv2d | nn.Linear, aug.model.get_submodule(n)).weight for n in producers]
            expected = sum(w.detach().abs().flatten(1).sum(dim=1) for w in weights)
            assert isinstance(expected, torch.Tensor)
            torch.testing.assert_close(imp[g], expected, rtol=1e-12, atol=0, msg=f"seed {seed} group {g}")
            groups += 1
            with_members += any(r.kind == "member" and g in r.in_groups for r in layers.values())
    assert groups > N_SEEDS // 2 and with_members >= 1, (groups, with_members)


def test_importance_ignores_extra_blocks_of_depth_augmentation():
    """A group that the original model has is ranked by layers outside the extra blocks
    only; a group that lives inside an extra block is ranked by that block's layers."""
    aug = NetAugModel(build("resnet18"), example_input(), width_mults=(1.0, 2.0), depth={"layer1": 1, "layer3": 1})
    randomize(aug, seed=21)
    extra = tuple(f"{p}.extra." for p in aug.max_depths)  # depth points: layer1.1, layer3.1
    kinds = aug.group_kinds()
    in_extra = {g for g, info in kinds.items() if info.block is not None and info.block.startswith(extra)}
    tied = {g for g, info in kinds.items() if info.kind == "depth-tied"}
    assert in_extra and tied, (in_extra, tied)

    for g, parts in aug.channel_importance(per_reader=True).items():
        if g in in_extra:
            assert parts and all(n.startswith(extra) for n in parts), (g, list(parts))
        else:
            assert not any(n.startswith(extra) for n in parts), (g, list(parts))

    before = aug.channel_importance()
    with torch.no_grad():  # change only the extra blocks' weights
        for n, p in aug.model.named_parameters():
            if n.startswith(extra):
                p.mul_(torch.rand_like(p) * 3)
    after = aug.channel_importance()
    for g in before:
        if g in in_extra:
            assert not torch.equal(after[g], before[g]), g
        else:
            assert torch.equal(after[g], before[g]), g


# --------------------------------------------------------------------------------------
# Fix 2: apply_permutations validates every entry before changing anything
# --------------------------------------------------------------------------------------
def _model_with_state() -> tuple[NetAugModel, torch.optim.Optimizer]:
    aug = wrap("conv_chain")
    randomize(aug, seed=3)
    aug.set_max()
    opt = torch.optim.SGD(aug.parameters(), lr=0.01, momentum=0.9)
    x = example_input()
    nn.functional.cross_entropy(aug(x), torch.tensor([0, 1])).backward()
    opt.step()
    return aug, opt


def _bad_entries(aug: NetAugModel) -> dict[str, tuple[object, object]]:
    g = max(aug.graph.free_groups(), key=lambda h: aug.max_sizes[h])
    n = aug.max_sizes[g]
    fixed = aug.graph.group_fixed.index(True)
    dup = torch.arange(n)
    dup[1] = 0
    neg = torch.arange(n)
    neg[0] = -1
    return {
        "unknown_group": (aug.graph.num_groups, torch.arange(n)),
        "negative_group": (-1, torch.arange(n)),
        "string_key": ("0", torch.arange(n)),
        "fixed_group": (fixed, torch.arange(aug.max_sizes[fixed])),
        "list_value": (g, list(range(n))),
        "float_tensor": (g, torch.arange(n, dtype=torch.float64)),
        "bool_tensor": (g, torch.ones(n, dtype=torch.bool)),
        "two_dim": (g, torch.arange(n).view(1, n)),
        "too_short": (g, torch.arange(n - 1)),
        "too_long": (g, torch.arange(n + 1)),
        "duplicate": (g, dup),
        "out_of_range": (g, neg),
    }


BAD_ENTRY_MESSAGES = {
    "unknown_group": "not a group id",
    "negative_group": "not a group id",
    "string_key": "not a group id",
    "fixed_group": "is fixed",
    "list_value": "1-D integer tensor",
    "float_tensor": "1-D integer tensor",
    "bool_tensor": "1-D integer tensor",
    "two_dim": "1-D integer tensor",
    "too_short": "entries, expected",
    "too_long": "entries, expected",
    "duplicate": "not a permutation",
    "out_of_range": "not a permutation",
}


@pytest.mark.parametrize("case", sorted(BAD_ENTRY_MESSAGES))
def test_apply_permutations_rejects_bad_entry(case):
    aug, opt = _model_with_state()
    key, value = _bad_entries(aug)[case]
    # a valid, non-identity entry first: it must not be applied either
    valid_g = min(aug.graph.free_groups(), key=lambda h: (key == h, -aug.max_sizes[h]))
    assert valid_g != key
    perms = {valid_g: torch.arange(aug.max_sizes[valid_g]).flip(0), key: value}
    snap = snapshot(aug, opt)
    with pytest.raises(ValueError, match=BAD_ENTRY_MESSAGES[case]):
        aug.apply_permutations(perms, opt)  # ty: ignore[invalid-argument-type]
    assert_unchanged(aug, opt, snap)


def test_apply_permutations_accepts_partial_and_int32():
    aug, opt = _model_with_state()
    g = aug.graph.free_groups()[0]
    before = aug.channel_importance()[g]
    p = torch.arange(aug.max_sizes[g], dtype=torch.int32).flip(0)
    aug.apply_permutations({g: p}, opt)
    torch.testing.assert_close(aug.channel_importance()[g], before.flip(0), rtol=1e-12, atol=0)


# --------------------------------------------------------------------------------------
# Fix 3: sort_channels checks extra_models before changing anything
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["other_width_mults", "other_architecture", "not_netaug"])
def test_sort_channels_rejects_mismatched_extra_model(case):
    aug, opt = _model_with_state()
    good_ema = copy.deepcopy(aug)
    if case == "other_width_mults":
        bad: object = wrap("conv_chain", mults=(1.0, 2.0))
    elif case == "other_architecture":
        bad = wrap("tiny_dense")
    else:
        bad = copy.deepcopy(aug.model)
    snaps = [snapshot(aug, opt), snapshot(good_ema)]
    bad_snap = snapshot(bad) if isinstance(bad, NetAugModel) else None
    with pytest.raises(ValueError, match="must be a NetAugModel" if case == "not_netaug" else "does not match"):
        aug.sort_channels(optimizer=opt, extra_models=[good_ema, bad])  # ty: ignore[invalid-argument-type]
    assert_unchanged(aug, opt, snaps[0])
    assert_unchanged(good_ema, None, snaps[1])
    if isinstance(bad, NetAugModel) and bad_snap is not None:
        assert_unchanged(bad, None, bad_snap)


def test_sort_channels_extra_models_generator():
    """A one-shot iterable still gets permuted after the check."""
    aug, _ = _model_with_state()
    ema = copy.deepcopy(aug)
    perms = aug.sort_channels(extra_models=(m for m in [ema]))
    assert any(not is_identity(p) for p in perms.values())
    for (n, p), (_, q) in zip(aug.state_dict().items(), ema.state_dict().items()):
        assert torch.equal(p, q), n


# --------------------------------------------------------------------------------------
# T-opt: sort-then-step == step-then-permute, for parameters, buffers and optimizer state
# --------------------------------------------------------------------------------------
OPTIMIZERS: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "sgd_nesterov": lambda ps: torch.optim.SGD(ps, lr=0.01, momentum=0.9, nesterov=True, weight_decay=5e-4),
    "adam": lambda ps: torch.optim.Adam(ps, lr=1e-3),
    "adamw": lambda ps: torch.optim.AdamW(ps, lr=1e-3, weight_decay=1e-2),
}
WARMUP = 3


def _cbr(c_in: int, c_out: int, k: int = 1, stride: int = 1, groups: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(c_in, c_out, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]


class _InvertedBlock(nn.Module):
    def __init__(self, c: int, e: int) -> None:
        super().__init__()
        self.body = nn.Sequential(*_cbr(c, e), *_cbr(e, e, 3, groups=e), *_cbr(e, c))

    def forward(self, x):
        return x + self.body(x)


class ResDW(nn.Module):
    """Residual inverted blocks with depthwise convs and batch norm, every BN followed by a ReLU.

    Not MobileNetV2: there, each project BN has no activation and feeds a 1x1 conv and a
    train-mode BN, which removes any per-channel shift, so the project BN biases have an
    exactly zero gradient. Their gradients are pure rounding noise, which Adam scales up to
    steps of size ``lr``, so the two branches differ by O(1) for a reason unrelated to the
    permutation. The same happens to a BN weight before ReLU + BN when its bias is exactly 0
    (default init), so ``_mild_init`` makes every BN bias nonzero.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_cbr(3, 8, 3), _InvertedBlock(8, 16), *_cbr(8, 12, 3, 2), _InvertedBlock(12, 24), _InvertedBlock(12, 20)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(12, 10)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.features(x)), 1))


@torch.no_grad()
def _mild_init(model: nn.Module, seed: int) -> None:
    """Distinct, non-degenerate BN values (weight in [0.5, 1.5), small nonzero bias)."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            assert m.running_mean is not None and m.running_var is not None
            m.weight.copy_(torch.rand(m.weight.shape, generator=g) + 0.5)
            m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
            m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
        elif isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
            m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)


def _batches(n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(123)
    return [(torch.randn(8, 3, 16, 16, generator=g), torch.randint(0, 10, (8,), generator=g)) for _ in range(n)]


def _step(m: NetAugModel, opt: torch.optim.Optimizer, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    opt.zero_grad()
    nn.functional.cross_entropy(m(batch[0]), batch[1]).backward()
    opt.step()


def _branch_deviation(opt_name: str, k: int, permute_state: bool) -> tuple[float, str, dict[int, torch.Tensor]]:
    """Run both branches and return (max relative deviation, worst tensor, perms)."""
    torch.manual_seed(0)
    # update_bn_in_aug=True: running statistics also move at max width, so buffers are tested
    aug = NetAugModel(ResDW(), torch.randn(2, 3, 16, 16), width_mults=(1.0, 2.0), update_bn_in_aug=True)
    _mild_init(aug, seed=21)
    aug.train()  # ResDW has no dropout
    aug.set_max()
    make = OPTIMIZERS[opt_name]
    opt0 = make(aug.parameters())
    batches = _batches(WARMUP + k)
    for b in batches[:WARMUP]:
        _step(aug, opt0, b)

    a, b = copy.deepcopy(aug), copy.deepcopy(aug)
    opt_a, opt_b = make(a.parameters()), make(b.parameters())
    opt_a.load_state_dict(copy.deepcopy(opt0.state_dict()))
    opt_b.load_state_dict(copy.deepcopy(opt0.state_dict()))

    perms = a.sort_channels(optimizer=opt_a if permute_state else None)
    for batch in batches[WARMUP:]:
        _step(a, opt_a, batch)
    for batch in batches[WARMUP:]:
        _step(b, opt_b, batch)
    b.apply_permutations(perms, opt_b if permute_state else None)

    worst, worst_name = 0.0, ""

    def check(name: str, x: torch.Tensor, y: torch.Tensor) -> None:
        nonlocal worst, worst_name
        d = rel_dev(x, y)
        if d > worst or not worst_name:
            worst, worst_name = d, name

    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    for n, p in pa.items():
        check(n, p, pb[n])
        sa, sb = opt_a.state[p], opt_b.state[pb[n]]
        assert sa.keys() == sb.keys() and sa, n
        for key, value in sa.items():
            if torch.is_tensor(value):
                check(f"{n}[{key}]", value, sb[key])
            else:
                assert value == sb[key], (n, key)
    ba, bb = dict(a.named_buffers()), dict(b.named_buffers())
    assert ba.keys() == bb.keys() and ba
    for n, buf in ba.items():
        check(n, buf, bb[n])
    return worst, worst_name, perms


@pytest.mark.parametrize("k", [1, 5, 20])
@pytest.mark.parametrize("opt_name", sorted(OPTIMIZERS))
def test_sort_commutes_with_optimizer_steps(opt_name, k):
    worst, name, perms = _branch_deviation(opt_name, k, permute_state=True)
    print(f"{opt_name} k={k}: max rel deviation {worst:.3g} ({name})")
    assert any(not is_identity(p) for p in perms.values()), "sorting did not move any channel"
    assert worst < OPT_REL, f"{name}: rel deviation {worst:.3g}"


@pytest.mark.parametrize("opt_name", ["sgd_nesterov", "adam"])
def test_sort_without_optimizer_state_breaks_commutation(opt_name):
    """Control: the same check fails when optimizer state is not permuted. Both halves are
    asserted together, so this test also fails if apply_permutations stops permuting state."""
    with_state, name, _ = _branch_deviation(opt_name, 5, permute_state=True)
    without_state, _, _ = _branch_deviation(opt_name, 5, permute_state=False)
    print(f"{opt_name}: with state {with_state:.3g}, without state {without_state:.3g}")
    assert with_state < OPT_REL, f"{name}: rel deviation {with_state:.3g}"
    assert without_state > 1e-6, without_state


# --------------------------------------------------------------------------------------
# .grad is permuted with its parameter
# --------------------------------------------------------------------------------------
def _resdw_with_state(
    opt_name: str,
) -> tuple[NetAugModel, torch.optim.Optimizer, tuple[torch.Tensor, torch.Tensor]]:
    """ResDW at max width after WARMUP optimizer steps (non-empty optimizer state)."""
    torch.manual_seed(0)
    aug = NetAugModel(ResDW(), torch.randn(2, 3, 16, 16), width_mults=(1.0, 2.0), update_bn_in_aug=True)
    _mild_init(aug, seed=21)
    aug.train()
    aug.set_max()
    opt = OPTIMIZERS[opt_name](aug.parameters())
    batches = _batches(WARMUP + 1)
    for b in batches[:WARMUP]:
        _step(aug, opt, b)
    return aug, opt, batches[WARMUP]


def _backward(m: NetAugModel, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    nn.functional.cross_entropy(m(batch[0]), batch[1]).backward()


def test_sort_permutes_grad():
    """backward -> sort gives the gradient of sort -> zero_grad -> backward."""
    aug, opt, batch = _resdw_with_state("sgd_nesterov")
    other = copy.deepcopy(aug)
    opt.zero_grad()
    _backward(aug, batch)
    perms = aug.sort_channels(optimizer=opt)
    assert any(not is_identity(p) for p in perms.values())

    other.apply_permutations(perms)
    other.zero_grad()
    _backward(other, batch)
    pa, pb = dict(aug.named_parameters()), dict(other.named_parameters())
    for n, p in pa.items():
        ga, gb = p.grad, pb[n].grad
        assert ga is not None and gb is not None, n
        d = rel_dev(ga, gb)
        assert d < OPT_REL, f"{n}.grad: rel deviation {d:.3g}"


@pytest.mark.parametrize("opt_name", ["sgd_nesterov", "adam"])
def test_sort_between_backward_and_step_commutes(opt_name):
    """backward -> sort(optimizer) -> step equals backward -> step -> apply the same perms."""
    aug, opt, batch = _resdw_with_state(opt_name)
    a, b = copy.deepcopy(aug), copy.deepcopy(aug)
    opt_a, opt_b = OPTIMIZERS[opt_name](a.parameters()), OPTIMIZERS[opt_name](b.parameters())
    opt_a.load_state_dict(copy.deepcopy(opt.state_dict()))
    opt_b.load_state_dict(copy.deepcopy(opt.state_dict()))
    for m, o in ((a, opt_a), (b, opt_b)):
        o.zero_grad()
        _backward(m, batch)

    perms = a.sort_channels(optimizer=opt_a)
    opt_a.step()
    opt_b.step()
    b.apply_permutations(perms, opt_b)
    assert any(not is_identity(p) for p in perms.values())

    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    for n, p in pa.items():
        d = rel_dev(p, pb[n])
        assert d < OPT_REL, f"{n}: rel deviation {d:.3g}"
        for key, value in opt_a.state[p].items():
            if torch.is_tensor(value):
                d = rel_dev(value, opt_b.state[pb[n]][key])
                assert d < OPT_REL, f"{n}[{key}]: rel deviation {d:.3g}"


def test_sort_permutes_grad_of_extra_models():
    aug, _, batch = _resdw_with_state("sgd_nesterov")
    ema = copy.deepcopy(aug)
    for m in (aug, ema):
        m.zero_grad()
        _backward(m, batch)
    aug.sort_channels(extra_models=[ema])
    fresh = copy.deepcopy(ema)  # sorted weights; its gradient is the reference
    fresh.zero_grad()
    _backward(fresh, batch)
    for (n, p), q in zip(ema.named_parameters(), fresh.parameters()):
        assert p.grad is not None and q.grad is not None
        d = rel_dev(p.grad, q.grad)
        assert d < OPT_REL, f"{n}.grad: rel deviation {d:.3g}"


def test_grad_none_stays_none():
    aug, opt, _ = _resdw_with_state("sgd_nesterov")
    opt.zero_grad(set_to_none=True)
    aug.sort_channels(optimizer=opt)
    assert all(p.grad is None for p in aug.parameters())


# --------------------------------------------------------------------------------------
# channel_importance(per_reader=True)
# --------------------------------------------------------------------------------------
def _check_per_reader(aug: NetAugModel) -> int:
    """Per-reader parts add up to the default importance; returns the number of parts."""
    total = aug.channel_importance()
    split = aug.channel_importance(per_reader=True)
    assert split.keys() == total.keys()
    order = list(aug.graph.layers)
    parts = 0
    for g, per in split.items():
        names = list(per)
        assert names == sorted(names, key=order.index), f"group {g}: not in graph.layers order"
        assert all(n in aug.graph.layers and aug.graph.layers[n].kind == "dense" for n in names)
        if not per:
            assert torch.equal(total[g], torch.zeros(aug.max_sizes[g]))
            continue
        parts += len(per)
        s = torch.stack(list(per.values())).sum(0)
        assert rel_dev(s, total[g]) <= 1e-12, f"group {g}"
    return parts


@pytest.mark.parametrize("name", ["mobilenet_v2", "resnet18", "tiny_dense", "efficientnet_b0"])
def test_per_reader_importance_adds_up(name):
    aug = wrap(name)
    randomize(aug, seed=9)
    assert _check_per_reader(aug) > len(aug.graph.free_groups())  # some group has several readers


def test_per_reader_importance_adds_up_on_random_architectures():
    for seed in range(0, N_SEEDS, 6):
        _check_per_reader(random_aug(seed))


class ReadTwice(nn.Module):
    """``b`` reads the same group twice through a concat."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Conv2d(3, 4, 1)
        self.b = nn.Conv2d(8, 5, 1)
        self.c = nn.Conv2d(4, 5, 1)

    def forward(self, x):
        y = torch.relu(self.a(x))
        return self.b(torch.cat([y, y], 1)) + self.c(y)


def test_per_reader_importance_adds_concat_segments_of_one_reader():
    aug = NetAugModel(randomize(ReadTwice()), torch.randn(2, 3, 4, 4))
    randomize(aug, seed=2)
    g = aug.graph.layers["a"].out_group
    assert g is not None and aug.graph.layers["b"].in_groups == (g, g)
    n = aug.max_sizes[g]
    per = aug.channel_importance(per_reader=True)[g]
    assert list(per) == ["b", "c"]
    wb, wc = (cast(nn.Conv2d, aug.model.get_submodule(name)).weight.detach() for name in "bc")
    expected_b = wb[:, :n].abs().sum(dim=(0, 2, 3)) + wb[:, n:].abs().sum(dim=(0, 2, 3))
    torch.testing.assert_close(per["b"], expected_b, rtol=1e-12, atol=0)
    torch.testing.assert_close(per["c"], wc.abs().sum(dim=(0, 2, 3)), rtol=1e-12, atol=0)


def test_per_reader_importance_fallback_uses_producers():
    """Same hand-written graph as the fallback test: group 1 has only a producer (and members),
    group 2 has neither readers nor producers."""
    graph = ChannelGraph(
        group_sizes=[3, 4, 5],
        group_fixed=[True, False, False],
        layers={
            "conv": LayerRole("dense", (0,), 1),
            "bn": LayerRole("member", (1,)),
            "dw": LayerRole("member", (1,)),
            "lonely_bn": LayerRole("member", (2,)),
        },
    )
    aug = NetAugModel(ProducerOnly(), graph=graph, width_mults=(1.0, 2.0))
    randomize(aug, seed=2)
    split = aug.channel_importance(per_reader=True)
    assert list(split[1]) == ["conv"] and split[2] == {}
    assert torch.equal(split[1]["conv"], aug.channel_importance()[1])
