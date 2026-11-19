"""Property test: for randomly generated architectures, sorting never changes the max-width
function, and base mode / export always reproduce the original model.

Each seed builds a random DAG from a mix of supported ops (dense conv, depthwise conv, BN,
activations, pooling, residual add, SE-style multiply, concat, linear head) and unsupported
ops (GroupNorm, grouped conv, layer-scale parameter, channel slice, channel shuffle). The
unsupported ops must make the analysis fix the touched groups, never make it wrong.
"""

import random

import pytest
import torch
import torch.nn as nn

from torchnetaug import NetAugModel

from conftest import assert_sorted, randomize

ACTS = [nn.ReLU, nn.SiLU, nn.Hardswish, nn.Sigmoid, nn.Tanh, nn.GELU, nn.ReLU6]
N_SEEDS = 60


class RandomNet(nn.Module):
    """A random DAG. ``self.plan`` is a fixed list, so torch.fx unrolls the loop."""

    def __init__(self, seed: int, in_ch: int = 3, n_ops: int = 14, n_classes: int = 5) -> None:
        super().__init__()
        rng = random.Random(seed)
        self.mods = nn.ModuleDict()
        self.params = nn.ParameterDict()
        self.plan = []
        self.used = []
        chans = [in_ch]  # channel count of each produced tensor
        pooled = set()  # tensors with 1x1 spatial size (inside SE branches): never picked

        def add_mod(m, src):
            name = f"m{len(self.mods)}"
            self.mods[name] = m
            self.plan.append(("mod", (src,), name))
            return name

        def pick():
            return rng.choice([i for i in range(len(chans)) if i not in pooled])

        def conv(src, out=None, k=None):
            out = out or rng.randint(2, 10)
            k = k or rng.choice([1, 3])
            add_mod(nn.Conv2d(chans[src], out, k, padding=k // 2, bias=rng.random() < 0.5), src)
            chans.append(out)
            return len(chans) - 1

        conv(0)  # make sure there is at least one free group
        for _ in range(n_ops):
            op = rng.choice(["conv", "conv", "dw", "bn", "act", "pool", "add", "add", "se", "cat",
                             "gn", "gconv", "scale", "slice", "shuffle"])
            src = pick()
            c = chans[src]
            if op == "conv":
                conv(src)
            elif op == "dw" and src != 0:
                add_mod(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=rng.random() < 0.5), src)
                chans.append(c)
            elif op == "bn" and src != 0:
                add_mod(nn.BatchNorm2d(c), src)
                chans.append(c)
            elif op == "act":
                add_mod(rng.choice(ACTS)(), src)
                chans.append(c)
            elif op == "pool":
                add_mod(rng.choice([nn.MaxPool2d(3, 1, 1), nn.AvgPool2d(3, 1, 1)]), src)
                chans.append(c)
            elif op == "add":
                same = [i for i, ci in enumerate(chans) if ci == c and i != src and i not in pooled]
                other = rng.choice(same) if same else conv(pick(), out=c)
                self.plan.append(("add", (src, other), None))
                chans.append(c)
            elif op == "se" and src != 0:
                add_mod(nn.AdaptiveAvgPool2d(1), src)
                chans.append(c)
                p = len(chans) - 1
                h = conv(p, out=max(1, c // 2), k=1)
                add_mod(nn.ReLU(), h)
                chans.append(chans[h])
                s = conv(len(chans) - 1, out=c, k=1)
                add_mod(nn.Sigmoid(), s)
                chans.append(c)
                pooled.update(range(p, len(chans)))
                self.plan.append(("mul", (src, len(chans) - 1), None))
                chans.append(c)
            elif op == "cat":
                k = rng.randint(2, 3)
                ins = tuple(pick() for _ in range(k))
                self.plan.append(("cat", ins, None))
                chans.append(sum(chans[i] for i in ins))
            elif op == "gn" and c % 2 == 0:
                add_mod(nn.GroupNorm(2, c), src)
                chans.append(c)
            elif op == "gconv" and c % 2 == 0:
                add_mod(nn.Conv2d(c, c, 3, padding=1, groups=2), src)
                chans.append(c)
            elif op == "scale":
                name = f"p{len(self.params)}"
                self.params[name] = nn.Parameter(torch.randn(1, c, 1, 1))
                self.plan.append(("scale", (src,), name))
                chans.append(c)
            elif op == "slice" and c >= 2:
                self.plan.append(("slice", (src,), None))
                chans.append(c // 2)
            elif op == "shuffle" and c % 2 == 0:
                self.plan.append(("shuffle", (src,), None))
                chans.append(c)

        # head: combine the last tensor with a random earlier one, pool, two linears
        last = max(i for i in range(len(chans)) if i not in pooled)
        if rng.random() < 0.5:
            other = pick()
            self.plan.append(("cat", (last, other), None))
            chans.append(chans[last] + chans[other])
            last = len(chans) - 1
        self.head_in = last
        self.fc1 = nn.Linear(chans[last], rng.randint(4, 12))
        self.fc2 = nn.Linear(self.fc1.out_features, n_classes)

    def forward(self, x):
        feats = [x]
        for kind, ins, name in self.plan:
            if kind == "mod":
                feats.append(self.mods[name](feats[ins[0]]))
            elif kind == "add":
                feats.append(feats[ins[0]] + feats[ins[1]])
            elif kind == "mul":
                feats.append(feats[ins[0]] * feats[ins[1]])
            elif kind == "cat":
                feats.append(torch.cat([feats[i] for i in ins], dim=1))
            elif kind == "scale":
                feats.append(feats[ins[0]] * self.params[name])
            elif kind == "slice":
                t = feats[ins[0]]
                feats.append(t[:, : t.shape[1] // 2])
            elif kind == "shuffle":
                t = feats[ins[0]]
                b, c, h, w = t.shape
                feats.append(t.view(b, 2, c // 2, h, w).transpose(1, 2).reshape(b, c, h, w))
        y = feats[self.head_in].mean([2, 3])
        return self.fc2(torch.relu(self.fc1(y)))


@pytest.mark.parametrize("seed", range(N_SEEDS))
def test_random_architecture(seed):
    torch.manual_seed(seed)
    base = randomize(RandomNet(seed), seed=seed)
    x = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(seed))
    aug = NetAugModel(base, x, width_mults=(1.0, 1.5, 2.5))

    for train_mode in (False, True):
        base.train(train_mode)
        aug.train(train_mode)
        aug.set_base()
        torch.testing.assert_close(aug(x), base(x), atol=1e-9, rtol=0, msg=f"base, train={train_mode}")

    randomize(aug, seed=seed + 1000)  # random values in the extra channels too
    for train_mode in (False, True):
        aug.train(train_mode)
        aug.set_max()
        before = aug(x)
        aug.sort_channels()  # importance order
        assert_sorted(aug)
        aug.set_max()
        torch.testing.assert_close(aug(x), before, atol=1e-8, rtol=1e-8, msg=f"sort, train={train_mode}")

        g = torch.Generator().manual_seed(seed)
        perms = {gid: torch.randperm(aug.max_sizes[gid], generator=g) for gid in aug.graph.free_groups()}
        aug.apply_permutations(perms)  # arbitrary order
        torch.testing.assert_close(aug(x), before, atol=1e-8, rtol=1e-8, msg=f"randperm, train={train_mode}")

    aug.eval()
    with aug.mode_ctx("base"):
        torch.testing.assert_close(aug.export().eval()(x), aug(x), atol=1e-9, rtol=0, msg="export")

    rng = random.Random(seed)
    for _ in range(3):
        aug.set_random(rng)
        assert aug(x).shape == (2, 5)


def test_random_architectures_exercise_the_analysis():
    """Guard against a generator that only produces trivial models."""
    free, fixed, members, concat_readers = 0, 0, 0, 0
    for seed in range(N_SEEDS):
        net = RandomNet(seed)
        aug = NetAugModel(net, torch.randn(1, 3, 8, 8))
        free += len(aug.graph.free_groups())
        fixed += aug.graph.num_groups - len(aug.graph.free_groups())
        members += sum(r.kind == "member" for r in aug.graph.layers.values())
        concat_readers += sum(len(r.in_groups) > 1 for r in aug.graph.layers.values())
    assert free > 3 * N_SEEDS and fixed > 3 * N_SEEDS
    assert members > N_SEEDS and concat_readers > N_SEEDS // 2
    print(f"free={free} fixed={fixed} members={members} concat_readers={concat_readers}")
