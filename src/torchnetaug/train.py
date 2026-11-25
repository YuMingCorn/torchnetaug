"""Training helpers: one NetAug step and batch-norm recalibration."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .layers import DynamicBatchNorm
from .netaug import NetAugModel

__all__ = ["netaug_step", "recalibrate_bn"]


def netaug_step(
    model: NetAugModel,
    inputs: Any,
    targets: Any,
    criterion: Callable[[Any, Any], torch.Tensor],
    alpha: float = 1.0,
    augment: bool = True,
    generator: random.Random | None = None,
    granularity: str = "group",
    backward: Callable[[torch.Tensor], None] = lambda loss: loss.backward(),
    forward: Callable[[nn.Module, Any], Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the forward/backward passes of one NetAug step (the optimizer step is yours).

    The objective is ``L(base) + alpha * L(augmented)`` with one sampled augmented model,
    as in the NetAug paper. The two backward passes accumulate gradients into the shared
    weights; call ``optimizer.step()`` and ``optimizer.zero_grad()`` afterwards.

    Args:
        backward: how to back-propagate a loss, e.g. ``lambda l: scaler.scale(l).backward()``
            for mixed precision.
        forward: optional ``forward(model, inputs) -> outputs`` (defaults to ``model(inputs)``).
        augment: set False to train only the base model (e.g. after augmentation stops).

    Returns ``(base_loss, aug_loss)`` detached; ``aug_loss`` is None if ``augment`` is False.
    """
    fwd = forward or (lambda m, x: m(x))
    inner = getattr(model, "module", model)  # DDP / DataParallel
    unwrapped = inner if isinstance(inner, NetAugModel) else model

    unwrapped.set_base()
    base_loss = criterion(fwd(model, inputs), targets)
    backward(base_loss)

    aug_loss = None
    if augment and alpha != 0:
        unwrapped.set_random(generator, granularity)
        aug_loss = criterion(fwd(model, inputs), targets)
        backward(alpha * aug_loss)
        unwrapped.set_base()
        aug_loss = aug_loss.detach()
    return base_loss.detach(), aug_loss


@torch.no_grad()
def recalibrate_bn(
    model: nn.Module,
    batches: Iterable[Any],
    num_batches: int | None = None,
    forward: Callable[[nn.Module, Any], Any] | None = None,
    sizes: Sequence[int] | None = None,
    depths: Mapping[str, int] | None = None,
) -> None:
    """Re-estimate batch-norm running statistics with exact averages over ``batches``.

    For a :class:`NetAugModel` this runs in base mode by default, so the statistics match
    the model that ``export()`` returns. Pass ``sizes`` (e.g. ``model.active_sizes`` after
    ``set_width(2.0)``) and/or ``depths`` (e.g. ``model.active_depths``) to calibrate another
    width or depth instead; what is not given stays at the base value. All widths share the
    same BN buffers, so calibrating one width overwrites the statistics of the others:
    evaluate other widths on a ``copy.deepcopy`` of the model. The previous width and depth
    are restored at the end.

    Each item of ``batches`` is passed to ``forward``; by default a ``(inputs, targets)``
    pair or a bare tensor is accepted.
    """
    inner = getattr(model, "module", model)
    netaug = inner if isinstance(inner, NetAugModel) else None
    if forward is None:

        def forward(m: nn.Module, batch: Any) -> Any:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            return m(x)

    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    saved = [(m, m.momentum) for m in bns]
    flags = [(m, m.training) for m in model.modules()]  # keep per-module flags (frozen BN, ...)
    previous = netaug.state() if netaug is not None else None
    try:
        if netaug is not None and previous is not None:
            netaug.set_base()
            if sizes is not None or depths is not None:
                netaug.set_state(
                    (
                        netaug.base_sizes if sizes is None else sizes,
                        {**netaug.base_depths, **(depths or {})},
                        "custom",
                    )
                )
            for m in bns:
                if isinstance(m, DynamicBatchNorm):
                    m.update_stats = True  # set_sizes() turns it off outside base mode
        idle = set()  # BNs of extra blocks that do not run keep their statistics
        if netaug is not None and previous is not None:
            for path in netaug._depth_paths:
                stack = netaug._stack(path)
                for extra in stack.extra[stack.active :]:
                    idle.update(id(m) for m in extra.modules())
        for m in bns:
            if m.track_running_stats and id(m) not in idle:
                m.reset_running_stats()
                m.momentum = None  # cumulative moving average
        model.train()
        for i, batch in enumerate(batches):
            if num_batches is not None and i >= num_batches:
                break
            forward(model, batch)
    finally:
        for m, momentum in saved:
            m.momentum = momentum
        for m, flag in flags:
            m.training = flag
        if netaug is not None and previous is not None:
            netaug.set_state(previous)
