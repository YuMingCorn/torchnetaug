import copy
from typing import Callable, Dict

import pytest
import torch
import torch.nn as nn
import torchvision

torch.set_default_dtype(torch.float64)


@torch.no_grad()
def randomize(model: nn.Module, seed: int = 0) -> nn.Module:
    """Give every parameter and BN buffer random values.

    Default BN values (weight 1, bias 0, mean 0, var 1) are the same for every channel,
    so a test could pass even if BN channels were never permuted. Random values catch that.
    """
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.3)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
            m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.3)
            m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
    return model


class ConvChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 12, 1), nn.BatchNorm2d(12), nn.ReLU6(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(12, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.pool(self.features(x))
        x = x.view(x.size(0), -1)
        return self.fc2(torch.relu(self.fc1(x)))


class TinyDense(nn.Module):
    """Concat along channels (DenseNet style), BN and depthwise conv after concat,
    and a residual add whose two sides are concats with the same layout."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 6, 3, padding=1)
        self.c1 = nn.Conv2d(6, 4, 3, padding=1)
        self.c2 = nn.Conv2d(10, 5, 3, padding=1)
        self.bn = nn.BatchNorm2d(15)
        self.dw = nn.Conv2d(15, 15, 3, padding=1, groups=15)
        self.head = nn.Linear(15, 3)

    def forward(self, x):
        x0 = torch.relu(self.stem(x))
        x1 = torch.relu(self.c1(x0))
        x01 = torch.cat([x0, x1], dim=1)
        x2 = torch.relu(self.c2(x01))
        y = torch.cat([x01, x2], 1)
        y = self.bn(y)
        y = self.dw(y) + y
        return self.head(torch.flatten(nn.functional.adaptive_avg_pool2d(y, 1), 1))


class ConcatMismatch(nn.Module):
    """Residual whose sides have different layouts (concat vs. one conv): must be fixed."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Conv2d(3, 4, 1)
        self.b = nn.Conv2d(3, 4, 1)
        self.proj = nn.Conv2d(8, 8, 1)
        self.c = nn.Conv2d(8, 6, 1)
        self.fc = nn.Linear(6, 2)

    def forward(self, x):
        y = torch.cat([self.a(x), self.b(x)], 1)
        y = self.proj(y) + y
        return self.fc(self.c(y).mean([2, 3]))


class WithGroupNorm(nn.Module):
    """GroupNorm and a layer-scale parameter are not whitelisted: their groups must be fixed."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.gn = nn.GroupNorm(2, 8)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.gamma = nn.Parameter(torch.ones(1, 8, 1, 1))
        self.c3 = nn.Conv2d(8, 6, 1)
        self.bn = nn.BatchNorm2d(6)
        self.fc = nn.Linear(6, 4)

    def forward(self, x):
        x = torch.relu(self.gn(self.c1(x)))
        x = self.c2(x) * self.gamma
        x = torch.relu(self.bn(self.c3(x)))
        return self.fc(x.mean([2, 3]))


MODELS: Dict[str, Callable[[], nn.Module]] = {
    "conv_chain": ConvChain,
    "tiny_dense": TinyDense,
    "group_norm": WithGroupNorm,
    "resnet18": lambda: torchvision.models.resnet18(num_classes=10),
    "mobilenet_v2": lambda: torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10),
    "mobilenet_v3_small": lambda: torchvision.models.mobilenet_v3_small(num_classes=10),
    "shufflenet_v2": lambda: torchvision.models.shufflenet_v2_x0_5(num_classes=10),
    "efficientnet_b0": lambda: torchvision.models.efficientnet_b0(num_classes=10),
}


@pytest.fixture(params=sorted(MODELS))
def model_name(request) -> str:
    return request.param


def example_input(batch: int = 2, size: int = 32, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, size, size, generator=g)


def build(name: str) -> nn.Module:
    torch.manual_seed(0)
    return randomize(MODELS[name]())


def clone(m):
    return copy.deepcopy(m)


def assert_sorted(aug, rtol: float = 1e-9) -> None:
    """After ``sort_channels()``: every free group is in non-increasing importance order,
    and sorting again only swaps channels whose importance is equal up to rounding.

    Recomputed importances are not bit-identical: permuting the *other* groups changes the
    order in which each reader's weights are summed. ``rtol`` covers that rounding (1e-9 is
    far above float64 rounding; use about 1e-5 for float32).
    """
    for g, imp in aug.channel_importance().items():
        ok = imp[:-1] >= imp[1:] - rtol * imp[1:].abs()
        assert torch.all(ok), f"group {g} not in non-increasing order: {imp.tolist()}"
    before = aug.channel_importance()
    for g, perm in aug.sort_channels().items():
        imp = before[g]
        torch.testing.assert_close(imp[perm], imp, rtol=rtol, atol=0,
                                   msg=f"second sort moved a channel of group {g} past a different importance")
