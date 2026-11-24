"""Dynamic layers whose weights are stored at the maximum width and sliced at run time.

Every dynamic layer reads its *input* channels through a "segment spec": a tuple of
``(offset, active)`` pairs. Each pair selects ``active`` channels starting at ``offset``
inside the max-width weight. A single segment at offset 0 is a plain prefix slice. More
than one segment happens after ``torch.cat``, where each concatenated tensor occupies its
own slot in the max-width layout.

Producer layers (dense conv / linear) also have ``active_out``: the number of output
channels, always a prefix of the max-width output dimension.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

Segments = tuple[tuple[int, int], ...]

__all__ = [
    "DynamicBatchNorm",
    "DynamicConv2d",
    "DynamicLinear",
    "DynamicModule",
    "segments_index",
]


def segments_index(segments: Segments, device: torch.device | None = None) -> torch.Tensor:
    """Return the flat channel index selected by ``segments``."""
    parts = [torch.arange(off, off + act, device=device) for off, act in segments]
    if not parts:
        return torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(parts)


def _is_prefix(segments: Segments) -> bool:
    return len(segments) == 1 and segments[0][0] == 0


class DynamicModule(nn.Module):
    """Mixin with the shared segment-selection logic."""

    in_segments: Segments
    active_out: int

    def _init_dynamic(self, in_size: int) -> None:
        self.in_segments = ((0, in_size),)
        self._index_cache: dict = {}

    def set_in_segments(self, segments: Sequence[tuple[int, int]]) -> None:
        self.in_segments = tuple((int(o), int(a)) for o, a in segments)
        self._index_cache = {}

    def _index(self, device: torch.device) -> torch.Tensor:
        key = (self.in_segments, device)
        idx = self._index_cache.get(key)
        if idx is None:
            idx = segments_index(self.in_segments, device)
            self._index_cache = {key: idx}
        return idx

    def _select_in(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Select the active input channels of ``t`` along ``dim``."""
        if _is_prefix(self.in_segments):
            return t.narrow(dim, 0, self.in_segments[0][1])
        return t.index_select(dim, self._index(t.device))

    @property
    def active_in(self) -> int:
        return sum(a for _, a in self.in_segments)


class DynamicConv2d(DynamicModule, nn.Conv2d):
    """``nn.Conv2d`` stored at max width.

    Two kinds are supported:

    * dense (``groups == 1``): output channels are a prefix of length ``active_out``,
      input channels follow ``in_segments``.
    * depthwise (``groups == in_channels == out_channels``): output channels equal the
      input channels, and both follow ``in_segments``.
    """

    def __init__(self, *args, **kwargs) -> None:
        nn.Conv2d.__init__(self, *args, **kwargs)
        if self.groups == 1:
            self.depthwise = False
        elif self.groups == self.in_channels == self.out_channels:
            self.depthwise = True
        else:
            raise ValueError("DynamicConv2d supports groups == 1 or depthwise convolution only")
        self._init_dynamic(self.in_channels)
        self.active_out = self.out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.depthwise:
            weight = self._select_in(self.weight, 0)
            bias = None if self.bias is None else self._select_in(self.bias, 0)
            groups = weight.shape[0]
        else:
            weight = self._select_in(self.weight[: self.active_out], 1)
            bias = None if self.bias is None else self.bias[: self.active_out]
            groups = 1
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding
        return F.conv2d(input, weight, bias, self.stride, padding, self.dilation, groups)

    def active_weight(self) -> torch.Tensor:
        if self.depthwise:
            return self._select_in(self.weight, 0)
        return self._select_in(self.weight[: self.active_out], 1)

    def active_bias(self) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self._select_in(self.bias, 0) if self.depthwise else self.bias[: self.active_out]


class DynamicLinear(DynamicModule, nn.Linear):
    """``nn.Linear`` stored at max width. Only 2-D inputs ``(batch, features)`` are analysed."""

    def __init__(self, *args, **kwargs) -> None:
        nn.Linear.__init__(self, *args, **kwargs)
        self._init_dynamic(self.in_features)
        self.active_out = self.out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.active_weight(), self.active_bias())

    def active_weight(self) -> torch.Tensor:
        return self._select_in(self.weight[: self.active_out], 1)

    def active_bias(self) -> torch.Tensor | None:
        return None if self.bias is None else self.bias[: self.active_out]


class DynamicBatchNorm(DynamicModule, nn.modules.batchnorm._BatchNorm):
    """Batch norm (1d/2d/3d) stored at max width; channels follow ``in_segments``.

    When ``update_stats`` is False and the module is training, the layer normalises with
    batch statistics but leaves ``running_mean`` / ``running_var`` untouched. NetAug uses
    this during the augmented forward pass so that the base model's statistics are not
    polluted by wider networks.
    """

    def __init__(self, *args, **kwargs) -> None:
        nn.modules.batchnorm._BatchNorm.__init__(self, *args, **kwargs)
        self._init_dynamic(self.num_features)
        self.update_stats = True

    def _check_input_dim(self, input: torch.Tensor) -> None:
        if input.dim() < 2:
            raise ValueError(f"expected at least 2D input (got {input.dim()}D input)")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        weight = None if self.weight is None else self._select_in(self.weight, 0)
        bias = None if self.bias is None else self._select_in(self.bias, 0)

        use_batch_stats = self.training or self.running_mean is None
        update = self.training and self.track_running_stats and self.update_stats

        momentum = 0.0 if self.momentum is None else self.momentum
        if update and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                momentum = 1.0 / float(self.num_batches_tracked)

        if use_batch_stats and not update:
            return F.batch_norm(input, None, None, weight, bias, True, 0.0, self.eps)

        assert self.running_mean is not None and self.running_var is not None
        prefix = _is_prefix(self.in_segments)
        mean = self._select_in(self.running_mean, 0)
        var = self._select_in(self.running_var, 0)
        if update and not prefix:
            # index_select returns a copy: update it, then write it back.
            mean, var = mean.clone(), var.clone()
        out = F.batch_norm(input, mean, var, weight, bias, use_batch_stats, momentum, self.eps)
        if update and not prefix:
            idx = self._index(input.device)
            self.running_mean.index_copy_(0, idx, mean)
            self.running_var.index_copy_(0, idx, var)
        return out
