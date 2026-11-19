"""The NetAug wrapper: builds the max-width model, switches widths, sorts channels, exports."""

from __future__ import annotations

import copy
import random
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .graph import ChannelGraph, analyze
from .layers import DynamicBatchNorm, DynamicConv2d, DynamicLinear, DynamicModule, segments_index

__all__ = ["NetAugModel"]

# (tensor name relative to the wrapped model, dim, offset, group id)
_PermOp = Tuple[str, int, int, int]


def _check_no_shared_parameters(model: nn.Module, graph: ChannelGraph) -> None:
    """Refuse tied weights on a resized layer: widening would silently untie them."""
    owners: Dict[int, List[str]] = {}
    for mod_name, mod in model.named_modules(remove_duplicate=False):
        for p_name, p in mod.named_parameters(recurse=False):
            owners.setdefault(id(p), []).append(f"{mod_name}.{p_name}" if mod_name else p_name)
    for names in owners.values():
        if len(names) > 1 and any(n.rsplit(".", 1)[0] in graph.layers for n in names):
            raise ValueError(
                "shared (tied) parameters are not supported on layers NetAug resizes: "
                + ", ".join(names)
            )


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


class NetAugModel(nn.Module):
    """Wrap a tiny model for Network Augmentation training.

    The wrapper copies ``model`` and widens every *free* channel group to
    ``round(size * max(width_mults))`` channels. The original weights are copied into the
    first channels, so in base mode the wrapper computes exactly what ``model`` computes.

    Args:
        model: the tiny (target) model. It is not modified.
        example_inputs: tensor or tuple of tensors used to trace shapes.
        width_mults: width multipliers relative to the base model. Must contain 1.0 as
            its smallest value, e.g. ``(1.0, 2.0, 3.0)``.
        graph: a pre-built :class:`ChannelGraph` (skips ``torch.fx`` analysis).
        update_bn_in_aug: if False (default), batch norm running statistics are only updated
            in base mode, so augmented passes cannot pollute the base model's statistics.

    Modes:
        * ``set_base()``: the original tiny model (used for evaluation and export).
        * ``set_max()``: every free group at its widest.
        * ``set_random()``: each free group samples a width from its choices.
    """

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Any = None,
        width_mults: Sequence[float] = (1.0, 2.0, 3.0),
        graph: Optional[ChannelGraph] = None,
        update_bn_in_aug: bool = False,
    ) -> None:
        super().__init__()
        mults = sorted(float(m) for m in width_mults)
        if not mults or mults[0] != 1.0:
            raise ValueError("width_mults must contain 1.0 as its smallest value")
        if graph is None:
            if example_inputs is None:
                raise ValueError("example_inputs is required when graph is not given")
            graph = analyze(model, example_inputs)
        _check_no_shared_parameters(model, graph)
        self.graph = graph
        self.width_mults = tuple(mults)
        self.update_bn_in_aug = update_bn_in_aug

        self.base_sizes: List[int] = list(graph.group_sizes)
        self.choices: List[List[int]] = []
        for size, fixed in zip(graph.group_sizes, graph.group_fixed):
            if fixed:
                self.choices.append([size])
            else:
                self.choices.append(sorted({max(1, round(size * m)) for m in mults}))
        self.max_sizes: List[int] = [c[-1] for c in self.choices]

        # Plain copy of the original model: export() loads sliced weights into a copy of it.
        self.__dict__["_template"] = copy.deepcopy(model)

        self.model = copy.deepcopy(model)
        self._in_offsets: Dict[str, List[int]] = {}
        for name, role in graph.layers.items():
            offsets, off = [], 0
            for g in role.in_groups:
                offsets.append(off)
                off += self.max_sizes[g]
            self._in_offsets[name] = offsets
            old = self.model.get_submodule(name)
            _set_submodule(self.model, name, self._widen(name, old))

        self.mode = "base"
        self.active_sizes: List[int] = list(self.base_sizes)
        self.set_base()

    # ------------------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------------------
    def _base_in_index(self, name: str, device: torch.device) -> torch.Tensor:
        role = self.graph.layers[name]
        segs = [(o, self.base_sizes[g]) for o, g in zip(self._in_offsets[name], role.in_groups)]
        return segments_index(tuple(segs), device)

    @torch.no_grad()
    def _widen(self, name: str, old: nn.Module) -> nn.Module:
        role = self.graph.layers[name]
        in_max = sum(self.max_sizes[g] for g in role.in_groups)
        p = next(old.parameters(), None)
        if p is None:
            p = next(old.buffers())
        fk = {"device": p.device, "dtype": p.dtype}
        idx = self._base_in_index(name, p.device)

        if isinstance(old, nn.Conv2d):
            common = dict(kernel_size=old.kernel_size, stride=old.stride, padding=old.padding,
                          dilation=old.dilation, bias=old.bias is not None,
                          padding_mode=old.padding_mode, **fk)
            if role.kind == "dense":
                out_max, out_base = self.max_sizes[role.out_group], self.base_sizes[role.out_group]
                new = DynamicConv2d(in_max, out_max, groups=1, **common)
                new.weight[:out_base].index_copy_(1, idx, old.weight)
                if old.bias is not None:
                    new.bias[:out_base].copy_(old.bias)
            else:
                new = DynamicConv2d(in_max, in_max, groups=in_max, **common)
                new.weight.index_copy_(0, idx, old.weight)
                if old.bias is not None:
                    new.bias.index_copy_(0, idx, old.bias)
        elif isinstance(old, nn.Linear):
            out_max, out_base = self.max_sizes[role.out_group], self.base_sizes[role.out_group]
            new = DynamicLinear(in_max, out_max, bias=old.bias is not None, **fk)
            new.weight[:out_base].index_copy_(1, idx, old.weight)
            if old.bias is not None:
                new.bias[:out_base].copy_(old.bias)
        elif isinstance(old, nn.modules.batchnorm._BatchNorm):
            new = DynamicBatchNorm(in_max, eps=old.eps, momentum=old.momentum, affine=old.affine,
                                   track_running_stats=old.track_running_stats, **fk)
            for attr in ("weight", "bias", "running_mean", "running_var"):
                src = getattr(old, attr)
                if src is not None:
                    getattr(new, attr).index_copy_(0, idx, src)
            if old.track_running_stats:
                new.num_batches_tracked.copy_(old.num_batches_tracked)
        else:  # pragma: no cover - the analysis only emits the types above
            raise TypeError(f"cannot widen {type(old).__name__}")
        for (n_old, p_old), (n_new, p_new) in zip(old.named_parameters(), new.named_parameters()):
            p_new.requires_grad_(p_old.requires_grad)
        new.train(old.training)
        return new

    def dynamic_layers(self) -> Iterator[Tuple[str, DynamicModule]]:
        for name in self.graph.layers:
            yield name, self.model.get_submodule(name)

    def num_active_params(self) -> int:
        """Number of parameters the current width uses.

        This equals the parameter count of a plain model built at the current sizes
        (for the base sizes: the parameter count of ``export()``).
        """
        dyn = dict(self.dynamic_layers())
        total = 0
        for name, mod in self.model.named_modules():
            if name in dyn:
                if isinstance(mod, (DynamicConv2d, DynamicLinear)):
                    total += mod.active_weight().numel()
                    total += 0 if mod.bias is None else mod.active_bias().numel()
                elif mod.affine:  # batch norm: one weight and one bias per channel
                    total += 2 * mod.active_in
            elif not any(name.startswith(d + ".") for d in dyn):
                total += sum(p.numel() for p in mod.parameters(recurse=False))
        return total

    # ------------------------------------------------------------------------------
    # width control
    # ------------------------------------------------------------------------------
    def set_sizes(self, sizes: Sequence[int], mode: str = "custom") -> None:
        """Set the active width of every channel group (one size per group)."""
        if len(sizes) != self.graph.num_groups:
            raise ValueError(f"expected {self.graph.num_groups} sizes, got {len(sizes)}")
        for g, s in enumerate(sizes):
            if not 1 <= s <= self.max_sizes[g]:
                raise ValueError(f"group {g}: size {s} outside [1, {self.max_sizes[g]}]")
            if self.graph.group_fixed[g] and s != self.base_sizes[g]:
                raise ValueError(f"group {g} is fixed at {self.base_sizes[g]} channels")
        self.active_sizes = [int(s) for s in sizes]
        self.mode = mode
        for name, layer in self.dynamic_layers():
            role = self.graph.layers[name]
            layer.set_in_segments(
                [(o, self.active_sizes[g]) for o, g in zip(self._in_offsets[name], role.in_groups)]
            )
            if role.kind == "dense":
                layer.active_out = self.active_sizes[role.out_group]
            if isinstance(layer, DynamicBatchNorm):
                layer.update_stats = mode == "base" or self.update_bn_in_aug

    def set_base(self) -> None:
        self.set_sizes(self.base_sizes, "base")

    def set_max(self) -> None:
        self.set_sizes(self.max_sizes, "max")

    def set_width(self, mult: float) -> None:
        """Scale every free group by the same multiplier (``mode == "custom"``).

        ``mult`` may lie between the training choices (e.g. 1.5), which is useful to check
        how accuracy changes with width. Sizes are clamped to ``[1, max_size]``.
        """
        sizes = [
            b if fixed else min(mx, max(1, round(b * mult)))
            for b, mx, fixed in zip(self.base_sizes, self.max_sizes, self.graph.group_fixed)
        ]
        self.set_sizes(sizes, "base" if sizes == self.base_sizes else "custom")

    def set_random(self, generator: Optional[random.Random] = None, granularity: str = "group") -> None:
        """Sample an augmented model.

        ``granularity="group"`` samples each free group on its own (a ResNet stage, an
        inverted-block expansion, ...). ``"global"`` samples one multiplier for the whole
        model. Pass a seeded ``random.Random`` to make sampling reproducible or to keep
        several distributed workers in sync.
        """
        rng = generator or random
        if granularity == "group":
            sizes = [rng.choice(c) for c in self.choices]
        elif granularity == "global":
            m = rng.choice(self.width_mults)
            sizes = [
                b if fixed else max(1, round(b * m))
                for b, fixed in zip(self.base_sizes, self.graph.group_fixed)
            ]
        else:
            raise ValueError("granularity must be 'group' or 'global'")
        self.set_sizes(sizes, "random")

    @contextmanager
    def mode_ctx(self, mode: str, **kwargs: Any) -> Iterator["NetAugModel"]:
        """Temporarily switch mode (``"base"``, ``"max"`` or ``"random"``)."""
        previous, prev_mode = list(self.active_sizes), self.mode
        getattr(self, f"set_{mode}")(**kwargs)
        try:
            yield self
        finally:
            self.set_sizes(previous, prev_mode)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------------------
    @torch.no_grad()
    def export(self, sizes: Optional[Sequence[int]] = None) -> nn.Module:
        """Return a plain PyTorch model holding the weights of one width.

        Without ``sizes`` (or with the base sizes) the result is a copy of the original
        model, same class and same shapes. With other sizes (for example
        ``model.export(model.active_sizes)`` after ``set_width(2.0)``) the result keeps the
        original model's structure and ``forward``, with every resized layer replaced by a
        plain ``nn.Conv2d`` / ``nn.Linear`` / batch-norm layer of the active size. Neither
        result depends on this package.
        """
        if sizes is not None and [int(s) for s in sizes] != self.base_sizes:
            return self._export_sizes(sizes)
        with self.mode_ctx("base"):
            state = self.model.state_dict()
            for name, layer in self.dynamic_layers():
                prefix = name + "."
                if isinstance(layer, (DynamicConv2d, DynamicLinear)):
                    state[prefix + "weight"] = layer.active_weight()
                    if layer.bias is not None:
                        state[prefix + "bias"] = layer.active_bias()
                else:
                    for attr in ("weight", "bias", "running_mean", "running_var"):
                        if getattr(layer, attr) is not None:
                            state[prefix + attr] = layer._select_in(getattr(layer, attr), 0)
            out = copy.deepcopy(self._template)
            cloned = OrderedDict((k, v.clone()) for k, v in state.items())
            # keep the per-module version info: some modules (e.g. MnasNet) require it
            cloned._metadata = getattr(state, "_metadata", None)  # type: ignore[attr-defined]
            out.load_state_dict(cloned)
        p = next(self.model.parameters(), None)
        if p is not None:
            out.to(p.device)
        out.train(self.training)
        return out

    def _export_sizes(self, sizes: Sequence[int]) -> nn.Module:
        previous, prev_mode = list(self.active_sizes), self.mode
        self.set_sizes(sizes)
        try:
            out = copy.deepcopy(self.model)
            for name, layer in self.dynamic_layers():
                _set_submodule(out, name, self._plain_layer(name, layer))
        finally:
            self.set_sizes(previous, prev_mode)
        out.train(self.training)
        return out

    def _plain_layer(self, name: str, layer: DynamicModule) -> nn.Module:
        """A plain layer of the original layer's class holding ``layer``'s active slice."""
        cls = type(self._template.get_submodule(name))
        n_in = layer.active_in
        tensors = list(layer.parameters()) + list(layer.buffers())
        device = tensors[0].device if tensors else torch.device("cpu")
        with torch.no_grad():
            if isinstance(layer, DynamicConv2d):
                n_out = n_in if layer.depthwise else layer.active_out
                new = cls(n_in, n_out, layer.kernel_size, layer.stride, layer.padding, layer.dilation,
                          groups=n_in if layer.depthwise else 1, bias=layer.bias is not None,
                          padding_mode=layer.padding_mode)
            elif isinstance(layer, DynamicLinear):
                new = cls(n_in, layer.active_out, bias=layer.bias is not None)
            else:
                new = cls(n_in, eps=layer.eps, momentum=layer.momentum, affine=layer.affine,
                          track_running_stats=layer.track_running_stats)
                for attr in ("running_mean", "running_var"):
                    if getattr(layer, attr) is not None:
                        getattr(new, attr).copy_(layer._select_in(getattr(layer, attr), 0))
                if layer.num_batches_tracked is not None:
                    new.num_batches_tracked.copy_(layer.num_batches_tracked)
                if layer.affine:
                    new.weight.copy_(layer._select_in(layer.weight, 0))
                    new.bias.copy_(layer._select_in(layer.bias, 0))
                return new.to(device)
            new.weight.copy_(layer.active_weight())
            if layer.bias is not None:
                new.bias.copy_(layer.active_bias())
        return new.to(device)

    # ------------------------------------------------------------------------------
    # channel sorting
    # ------------------------------------------------------------------------------
    def _perm_ops(self) -> Dict[int, List[_PermOp]]:
        """For each free group, list every tensor slice that carries its channels."""
        ops: Dict[int, List[_PermOp]] = {g: [] for g in self.graph.free_groups()}
        for name, layer in self.dynamic_layers():
            role = self.graph.layers[name]
            if role.kind == "dense":
                if role.out_group in ops:
                    ops[role.out_group].append((f"{name}.weight", 0, 0, role.out_group))
                    if layer.bias is not None:
                        ops[role.out_group].append((f"{name}.bias", 0, 0, role.out_group))
                for off, g in zip(self._in_offsets[name], role.in_groups):
                    if g in ops:
                        ops[g].append((f"{name}.weight", 1, off, g))
            else:
                attrs = ["weight", "bias"]
                if isinstance(layer, DynamicBatchNorm):
                    attrs += ["running_mean", "running_var"]
                for off, g in zip(self._in_offsets[name], role.in_groups):
                    if g in ops:
                        for attr in attrs:
                            if getattr(layer, attr, None) is not None:
                                ops[g].append((f"{name}.{attr}", 0, off, g))
        return ops

    @torch.no_grad()
    def channel_importance(self) -> Dict[int, torch.Tensor]:
        """L1 importance of each channel of each free group (max-width layout).

        A channel is important if the layers that *read* it give it large weights. If a
        group has no dense reader, the producers' output weights are used instead.
        """
        tensors = dict(self.model.named_parameters())
        result: Dict[int, torch.Tensor] = {}
        for g, ops in self._perm_ops().items():
            size = self.max_sizes[g]
            readers = [(n, d, o) for n, d, o, _ in ops if d == 1]
            if not readers:
                readers = [(n, d, o) for n, d, o, _ in ops if d == 0 and n.endswith(".weight")]
            imp = None
            for n, d, o in readers:
                w = tensors[n].detach().narrow(d, o, size).abs()
                w = w.to(torch.promote_types(w.dtype, torch.float32))
                dims = [i for i in range(w.dim()) if i != d]
                v = w.sum(dim=dims) if dims else w
                imp = v if imp is None else imp + v
            result[g] = imp if imp is not None else torch.zeros(size)
        return result

    @torch.no_grad()
    def sort_channels(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_models: Iterable["NetAugModel"] = (),
    ) -> Dict[int, torch.Tensor]:
        """Reorder channels so the most important ones come first.

        Every free group gets one permutation, applied to every tensor that carries the
        group's channels: producer outputs, batch-norm/depthwise channels, and the matching
        input slice of every reader (including concat offsets and residual branches). The
        max-width function is therefore unchanged; only which channels the smaller widths
        keep changes.

        ``optimizer`` state tensors with the same shape as a permuted parameter (momentum,
        Adam moments, ...) get the same permutation. ``extra_models`` (e.g. an EMA copy with
        the same structure) are permuted with the same permutations.

        Returns ``{group_id: permutation}``.
        """
        importance = self.channel_importance()
        perms = {g: torch.argsort(imp, descending=True, stable=True) for g, imp in importance.items()}
        self.apply_permutations(perms, optimizer)
        for other in extra_models:
            other.apply_permutations(perms)
        return perms

    @torch.no_grad()
    def apply_permutations(
        self, perms: Dict[int, torch.Tensor], optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """Apply ``{group_id: permutation}``: new channel ``i`` = old channel ``perm[i]``."""
        tensors: Dict[str, torch.Tensor] = dict(self.model.named_parameters())
        tensors.update(dict(self.model.named_buffers()))
        params = dict(self.model.named_parameters())
        for g, ops in self._perm_ops().items():
            if g not in perms:
                continue
            perm = perms[g]
            for name, dim, off, _ in ops:
                targets = [tensors[name]]
                if optimizer is not None and name in params:
                    state = optimizer.state.get(params[name], {})
                    targets += [v for v in state.values()
                                if torch.is_tensor(v) and v.shape == params[name].shape]
                for t in targets:
                    idx = torch.arange(off, off + perm.numel(), device=t.device)
                    src = t.index_select(dim, idx[perm.to(t.device)])
                    t.index_copy_(dim, idx, src)
