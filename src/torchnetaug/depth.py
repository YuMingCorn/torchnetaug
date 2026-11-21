"""Depth augmentation: extra copies of a block that only the augmented models run.

NetAug (the paper and its official code) augments width only. Depth augmentation here
follows the *elastic depth* idea of Once-for-All (Cai et al., ICLR 2020, same group): a
stage keeps its first blocks and may run more. For NetAug, the base model is the
original network, and an augmented model runs ``k`` extra blocks after a chosen block.

* An extra block is a copy of the chosen block with freshly initialized weights. When the
  block is residual, the last batch norm of each copy starts at zero, so a copy starts as
  the identity and the deeper network starts with the base model's function.
* The chosen block must keep the shape of its input (same channels and spatial size),
  because its copies read its output.
* The output of every extra block is tied to the output of the chosen block: they are
  one channel group, so they always have the same width and the same channel order. Then
  the layer after the stack reads the same channels whether the extras run or not, and
  channel sorting stays consistent at every depth.

Users do not build :class:`DepthStack` by hand: pass ``depth={...}`` to ``NetAugModel``.
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn

__all__ = ["DepthStack", "tie_channels"]


def tie_channels(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Return ``x`` unchanged. The channel analysis puts ``x`` and ``ref`` in one group."""
    return x


fx.wrap("tie_channels")  # keep the call as one node in the traced graph


class DepthStack(nn.Module):
    """A block followed by extra copies of it; ``active`` copies run in ``forward``.

    ``active == 0`` computes exactly ``block(x)``.
    """

    def __init__(self, block: nn.Module, extras: List[nn.Module]) -> None:
        super().__init__()
        self.block = block
        self.extra = nn.ModuleList(extras)
        self.active = len(extras)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        for i in range(self.active):
            y = tie_channels(self.extra[i](y), y)
        return y

    def to_plain(self) -> nn.Module:
        """The active part as plain modules: ``block`` alone, or ``nn.Sequential``."""
        if self.active == 0:
            return self.block
        seq = nn.Sequential(self.block, *list(self.extra)[: self.active])
        seq.training = self.training
        return seq


def _reset(module: nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
            m.reset_running_stats()


@torch.no_grad()
def make_extra(block: nn.Module, example: Optional[torch.Tensor]) -> Tuple[nn.Module, Optional[str]]:
    """A copy of ``block`` with new weights; identity at start when ``block`` is residual.

    The last batch norm of the copy gets weight 0. If the copy then returns its input
    unchanged on ``example`` (or a positive non-zero probe when ``example`` is near zero,
    so a zero input cannot mistake a plain block for a residual one), it is a residual
    block and the copy is kept. Otherwise a plain re-initialized copy is used.

    Returns ``(extra, zeroed_bn_path)`` where ``zeroed_bn_path`` is the relative module
    name of the zeroed BN (or ``None`` when the copy was not zeroed).
    """
    extra = copy.deepcopy(block)
    _reset(extra)
    norms = [(n, m) for n, m in extra.named_modules()
             if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.affine]
    if not norms or example is None or example.numel() == 0:
        return extra, None
    probe = example
    if probe.abs().max().item() < 1e-3:
        # A zero input (e.g. tracing with torch.zeros) cannot distinguish f(x)=x from f(x)=0.
        # Use a positive probe so post-ReLU residual blocks (BasicBlock) still pass f(x)=x.
        g = torch.Generator(device="cpu").manual_seed(0)
        probe = torch.rand(example.shape, generator=g, dtype=torch.float32).to(
            device=example.device, dtype=example.dtype
        ) + 0.5
    candidate = copy.deepcopy(extra)
    last_name, last = [(n, m) for n, m in candidate.named_modules()
                       if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.affine][-1]
    last.weight.zero_()
    last.bias.zero_()
    flags = [(m, m.training) for m in candidate.modules()]  # keep per-module flags
    candidate.eval()
    out: Any = candidate(probe)
    for m, flag in flags:
        m.training = flag
    if isinstance(out, torch.Tensor) and out.shape == probe.shape and torch.allclose(
        out.float(), probe.float(), rtol=1e-5, atol=1e-6
    ):
        return candidate, last_name
    return extra, None
