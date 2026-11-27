"""Group classification (``group_kinds``) and ``sort_channels(scope=..., blocks=...)``."""

import copy
from collections import Counter

import pytest
import torch
import torchvision
from conftest import assert_sorted, build, example_input, randomize
from torch import nn
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.models.resnet import BasicBlock

from torchnetaug import ChannelGraph, GroupInfo, LayerRole, NetAugModel

tv = torchvision.models


def _x() -> torch.Tensor:
    return torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(1))


def _wrap(model: nn.Module, **kw) -> NetAugModel:
    return NetAugModel(model, _x(), **kw)


def _counts(kinds: dict[int, GroupInfo]) -> dict[str, int]:
    return dict(Counter(i.kind for i in kinds.values()))


def _of_kind(kinds: dict[int, GroupInfo], kind: str) -> list[GroupInfo]:
    return [i for i in kinds.values() if i.kind == kind]


# --------------------------------------------------------------------------------------
# automatic blocks on real models (the table of the design)
# --------------------------------------------------------------------------------------
class Backbone(nn.Module):
    """Not a container: the ResNet-18 stem and its 4 stages."""

    def __init__(self) -> None:
        super().__init__()
        r = tv.resnet18(num_classes=10)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class BackboneHead(nn.Module):
    """A non-container backbone and a head next to it."""

    def __init__(self, hidden: int = 0) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.fc = (
            nn.Sequential(nn.Linear(512, hidden), nn.ReLU(), nn.Linear(hidden, 10)) if hidden else nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(self.backbone(x), 1), 1))


class Wrapper(nn.Module):
    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def test_mobilenet_v2_kinds():
    aug = _wrap(tv.mobilenet_v2(num_classes=10))
    kinds = aug.group_kinds()
    assert _counts(kinds) == {"inner": 16, "residual": 5, "closed": 4, "fixed": 2}
    inner = _of_kind(kinds, "inner")
    assert all(len(i.modules) == 5 for i in inner)
    assert sorted(i.block or "" for i in inner) == sorted(f"features.{k}" for k in range(2, 18))
    expand = [n for n in aug.graph.layers if n.endswith(".conv.0.0") and aug.graph.layers[n].kind == "dense"]
    assert len(expand) == 16
    for n in expand:
        g = aug.graph.layers[n].out_group
        assert g is not None
        info = kinds[g]
        assert info.kind == "inner" and len(info.modules) == 5, (n, info)
        block = n.removesuffix(".conv.0.0")
        assert info.block == block
        assert info.producers == (n,) and info.readers == (f"{block}.conv.2",)
        assert info.members == (f"{block}.conv.0.1", f"{block}.conv.1.0", f"{block}.conv.1.1")
        assert info.modules == (n, *info.members, f"{block}.conv.2")


def test_resnet18_kinds():
    aug = _wrap(tv.resnet18(num_classes=10))
    kinds = aug.group_kinds()
    assert _counts(kinds) == {"inner": 8, "residual": 4, "fixed": 2}
    assert all(len(i.modules) == 3 for i in _of_kind(kinds, "inner"))
    for s in range(1, 5):
        for b in range(2):
            g = aug.graph.layers[f"layer{s}.{b}.conv1"].out_group
            assert g is not None
            assert kinds[g].kind == "inner" and kinds[g].block == f"layer{s}.{b}"
            assert kinds[g].modules == (f"layer{s}.{b}.conv1", f"layer{s}.{b}.bn1", f"layer{s}.{b}.conv2")
        trunk = aug.graph.layers[f"layer{s}.1.conv2"].out_group
        assert trunk is not None and kinds[trunk].kind == "residual"
        assert len(kinds[trunk].producers) > 1


def test_resnet50_kinds():
    kinds = _wrap(tv.resnet50(num_classes=10)).group_kinds()
    assert _counts(kinds) == {"inner": 32, "residual": 4, "closed": 1, "fixed": 2}


def test_mobilenet_v3_small_kinds():
    kinds = _wrap(tv.mobilenet_v3_small(num_classes=10)).group_kinds()
    assert _counts(kinds) == {"inner": 19, "residual": 5, "closed": 3, "fixed": 2}
    assert dict(Counter(len(i.modules) for i in _of_kind(kinds, "inner"))) == {2: 9, 5: 2, 7: 8}


def test_resnet18_with_depth_kinds():
    aug = _wrap(tv.resnet18(num_classes=10), depth={"layer1": 1, "layer3": 1})
    kinds = aug.group_kinds()
    assert _counts(kinds) == {"depth-tied": 2, "inner": 10, "residual": 2, "fixed": 2}
    blocks = {i.block for i in _of_kind(kinds, "inner")}
    assert {"layer1.1.extra.0", "layer3.1.extra.0"} <= blocks
    for info in _of_kind(kinds, "depth-tied"):
        assert any(n.startswith(("layer1.1.extra.0.", "layer3.1.extra.0.")) for n in info.modules)


@pytest.mark.parametrize(("hidden", "closed"), [(0, 0), (64, 1)])
def test_backbone_module_is_not_a_block(hidden, closed):
    """A non-container module holding the whole trunk must not become one block. With one
    ``fc`` there is no closed group; a two-layer head adds one (its hidden group)."""
    kinds = _wrap(BackboneHead(hidden)).group_kinds()
    expected = {"inner": 8, "residual": 4, "fixed": 2}
    if closed:
        expected["closed"] = closed
    assert _counts(kinds) == expected
    assert all(i.block != "backbone" for i in kinds.values())


def test_wrapper_module_changes_nothing():
    plain = _wrap(tv.mobilenet_v2(num_classes=10)).group_kinds()
    wrapped = _wrap(Wrapper(tv.mobilenet_v2(num_classes=10))).group_kinds()
    assert [(i.group, i.kind, i.block) for i in plain.values()] == [
        (i.group, i.kind, None if i.block is None else i.block.removeprefix("m.")) for i in wrapped.values()
    ]
    assert [i.modules for i in plain.values()] == [
        tuple(n.removeprefix("m.") for n in i.modules) for i in wrapped.values()
    ]


# --------------------------------------------------------------------------------------
# ChannelGraph.group_kinds: only names and roles
# --------------------------------------------------------------------------------------
def _hand_graph() -> ChannelGraph:
    """g0 input (fixed); g1, g2 read as a concat by ``outer.blk.b``; g3 one producer;
    g4 a residual through the extra block ``s.extra.0``; g5 output (fixed)."""
    return ChannelGraph(
        group_sizes=[3, 4, 4, 6, 8, 2],
        group_fixed=[True, False, False, False, False, True],
        layers={
            "outer.blk.a": LayerRole("dense", (0,), 1),
            "outer.blk.c": LayerRole("dense", (0,), 2),
            "outer.blk.bn": LayerRole("member", (2,)),
            "outer.blk.b": LayerRole("dense", (1, 2), 3),
            "s.block.conv": LayerRole("dense", (3,), 4),
            "s.extra.0.conv": LayerRole("dense", (4,), 4),
            "head": LayerRole("dense", (4,), 5),
        },
    )


def test_graph_kinds_without_blocks():
    kinds = _hand_graph().group_kinds()
    assert list(kinds) == list(range(6))
    assert [i.kind for i in kinds.values()] == [
        "fixed",
        "concat-segment",
        "concat-segment",
        "closed",
        "residual",
        "fixed",
    ]
    assert all(i.block is None for i in kinds.values())


def test_graph_kinds_precedence():
    # inner beats concat-segment; depth-tied beats residual; the innermost block wins
    kinds = _hand_graph().group_kinds(blocks=["outer", "outer.blk", "outer.bl", "s"], extras=["s.extra.0"])
    assert [(i.kind, i.block) for i in kinds.values()] == [
        ("fixed", None),
        ("inner", "outer.blk"),
        ("inner", "outer.blk"),
        ("closed", None),
        ("depth-tied", None),
        ("fixed", None),
    ]
    g2, g4 = kinds[2], kinds[4]
    assert g2.producers == ("outer.blk.c",) and g2.members == ("outer.blk.bn",) and g2.readers == ("outer.blk.b",)
    assert g2.modules == ("outer.blk.c", "outer.blk.bn", "outer.blk.b")
    assert g4.producers == ("s.block.conv", "s.extra.0.conv")
    assert g4.readers == ("s.extra.0.conv", "head")
    assert g4.modules == ("s.block.conv", "s.extra.0.conv", "head")
    assert kinds[0].modules == () and kinds[5].modules == ()
    # without the extra block, g4 is a residual group (two producers)
    assert _hand_graph().group_kinds(blocks=["outer.blk"])[4].kind == "residual"


def test_graph_kinds_prefix_needs_a_dot():
    kinds = _hand_graph().group_kinds(blocks=["outer.bl", "outer.blk.a"])
    assert all(i.kind != "inner" for i in kinds.values())


# --------------------------------------------------------------------------------------
# blocks given by the user
# --------------------------------------------------------------------------------------
def test_blocks_by_path_and_by_class():
    aug = NetAugModel(build("mobilenet_v2"), example_input())
    auto = {g for g, i in aug.group_kinds().items() if i.kind == "inner"}
    by_class = aug.group_kinds(blocks=[InvertedResidual])
    assert {g for g, i in by_class.items() if i.kind == "inner"} == auto
    by_path = aug.group_kinds(blocks=["features.3"])
    inner = _of_kind(by_path, "inner")
    assert [i.block for i in inner] == ["features.3"]
    perms = aug.sort_channels(scope="inner", blocks=["features.3"])
    assert set(perms) == {inner[0].group}


def test_blocks_class_without_instances_gives_no_inner_group():
    aug = NetAugModel(build("mobilenet_v2"), example_input())
    assert not _of_kind(aug.group_kinds(blocks=[BasicBlock]), "inner")


def test_blocks_bad_path():
    aug = NetAugModel(build("mobilenet_v2"), example_input())
    with pytest.raises(ValueError, match="not a module path"):
        aug.group_kinds(blocks=["features.99"])
    with pytest.raises(ValueError, match="not a module path"):
        aug.sort_channels(scope="inner", blocks=["nope"])


# --------------------------------------------------------------------------------------
# sort_channels(scope=...)
# --------------------------------------------------------------------------------------
def _state(aug: NetAugModel) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in aug.state_dict().items()}


def _mbv2() -> NetAugModel:
    aug = NetAugModel(build("mobilenet_v2"), example_input())
    randomize(aug, seed=11)
    return aug


def test_scope_keys():
    aug = _mbv2()
    kinds = aug.group_kinds()
    inner = {g for g, i in kinds.items() if i.kind == "inner"}
    closed = {g for g, i in kinds.items() if i.kind == "closed"}
    assert inner and closed
    assert set(copy.deepcopy(aug).sort_channels(scope="inner")) == inner
    assert set(copy.deepcopy(aug).sort_channels(scope="closed")) == inner | closed
    assert set(copy.deepcopy(aug).sort_channels(scope="all")) == set(aug.graph.free_groups())
    assert set(copy.deepcopy(aug).sort_channels(scope=sorted(closed))) == closed
    assert copy.deepcopy(aug).sort_channels(scope=set()) == {}


@pytest.mark.parametrize("name", ["mobilenet_v2", "resnet18", "tiny_dense", "shufflenet_v2"])
def test_default_scope_is_the_old_behaviour(name):
    """sort_channels() == sort_channels(scope="all") == argsort of the importance applied
    with apply_permutations (the code path before scopes existed), bit for bit."""
    aug = NetAugModel(build(name), example_input())
    randomize(aug, seed=12)
    ref = copy.deepcopy(aug)
    pref = {g: torch.argsort(imp, descending=True, stable=True) for g, imp in ref.channel_importance().items()}
    ref.apply_permutations(pref)
    sref = ref.state_dict()
    for scope in (None, "all"):  # one copy at a time: float64 resnet18 copies are large
        m = copy.deepcopy(aug)
        perms = m.sort_channels() if scope is None else m.sort_channels(scope=scope)
        assert perms.keys() == pref.keys()
        assert all(torch.equal(perms[g], pref[g]) for g in perms)
        sm = m.state_dict()
        assert sm.keys() == sref.keys()
        assert all(torch.equal(sm[k], sref[k]) for k in sm)
        del m, sm


@pytest.mark.parametrize(
    ("scope", "match"),
    [
        ("everything", "scope must be one of"),
        ("", "scope must be one of"),
        (3, "scope must be one of"),
        ([10_000], "not a group id"),
        (["a"], "not a group id"),
        ([True], "not a group id"),
    ],
)
def test_bad_scope(scope, match):
    aug = _mbv2()
    opt = torch.optim.SGD(aug.parameters(), lr=0.1, momentum=0.9)
    before = _state(aug)
    with pytest.raises(ValueError, match=match):
        aug.sort_channels(opt, scope=scope)
    after = _state(aug)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_scope_with_fixed_group_raises_and_returns_never_contain_fixed():
    aug = _mbv2()
    fixed = [g for g, f in enumerate(aug.graph.group_fixed) if f]
    free = aug.graph.free_groups()
    assert fixed
    before = _state(aug)
    with pytest.raises(ValueError, match="is fixed"):
        aug.sort_channels(scope=[free[0], fixed[0]])
    after = _state(aug)
    assert all(torch.equal(before[k], after[k]) for k in before)
    for scope in ("all", "closed", "inner", free):
        assert not set(copy.deepcopy(aug).sort_channels(scope=scope)) & set(fixed), scope


# --------------------------------------------------------------------------------------
# V3: sorting one kind of group keeps the max-width function and touches nothing else
# --------------------------------------------------------------------------------------
def _kind_model(kind: str) -> NetAugModel:
    torch.manual_seed(0)
    if kind == "concat-segment":
        aug = NetAugModel(build("tiny_dense"), example_input())
    elif kind == "depth-tied":
        aug = NetAugModel(randomize(tv.resnet18(num_classes=10)), example_input(), depth={"layer1": 1, "layer3": 1})
    else:
        aug = NetAugModel(build("mobilenet_v2"), example_input())
    randomize(aug, seed=5)  # also the extra channels and the extra blocks
    for m in aug.modules():  # a dropout mask over channels is not permutation invariant
        if isinstance(m, nn.modules.dropout._DropoutNd):
            m.p = 0.0
    return aug


def _outputs(aug: NetAugModel, x: torch.Tensor) -> list[torch.Tensor]:
    """Max width in eval and train mode (and at depth 0 if the model has depth points)."""
    outs = []
    states = [(aug.max_sizes, aug.max_depths)]
    if aug.max_depths:
        states.append((aug.max_sizes, aug.base_depths))
    with torch.no_grad():
        for train in (False, True):
            aug.train(train)
            for sizes, depths in states:
                aug.set_state((sizes, depths, "max"))
                outs.append(aug(x))
    aug.eval()
    aug.set_base()
    return outs


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max()).item()


@pytest.mark.parametrize("kind", ["inner", "residual", "closed", "concat-segment", "depth-tied"])
def test_sort_one_kind(kind):
    aug = _kind_model(kind)
    kinds = aug.group_kinds()
    groups = {g for g, i in kinds.items() if i.kind == kind}
    assert groups, f"the model has no {kind} group"
    x = example_input()
    before_out = _outputs(aug, x)
    with torch.no_grad(), aug.mode_ctx("base"):
        before_base = aug(x)
    before = _state(aug)

    perms = aug.sort_channels(scope=groups)
    assert set(perms) == groups
    assert any(not torch.equal(p, torch.arange(p.numel())) for p in perms.values())

    # Tensors of layers that no sorted group touches are bit-identical ...
    after = _state(aug)
    touched = {n for g in groups for n in kinds[g].modules}
    for k in before:
        if k.removeprefix("model.").rsplit(".", 1)[0] not in touched:
            assert torch.equal(before[k], after[k]), k
    # ... and undoing the sorted groups' permutations restores every tensor bit for bit,
    # so the other groups' channels did not move inside the shared tensors either.
    undo = copy.deepcopy(aug)
    undo.apply_permutations({g: torch.argsort(p) for g, p in perms.items()})
    restored = _state(undo)
    assert all(torch.equal(before[k], restored[k]) for k in before)

    # _outputs gives the eval-mode outputs first, then the train-mode ones. In train mode,
    # batch norm normalizes over 2 values per channel in the last layers (some with zero
    # variance), which amplifies float64 rounding to about 1e-10 with any thread count.
    n_eval = len(before_out) // 2
    for i, (b, a) in enumerate(zip(before_out, _outputs(aug, x))):
        assert _rel(a, b) < (1e-10 if i < n_eval else 1e-8), f"output {i}"
    assert_sorted(aug, groups=groups)
    if kind == "inner":
        with torch.no_grad(), aug.mode_ctx("base"):
            assert (aug(x) - before_base).abs().max() > 1e-6
