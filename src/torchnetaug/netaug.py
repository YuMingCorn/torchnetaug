"""The NetAug wrapper: builds the max-width model, switches widths, sorts channels, exports."""

from __future__ import annotations

import copy
import random
from collections import OrderedDict
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import nn

from .depth import DepthStack, make_extra
from .graph import ChannelGraph, analyze, check_plain
from .layers import DynamicBatchNorm, DynamicConv2d, DynamicLinear, DynamicModule, segments_index

__all__ = ["NetAugModel"]

# (tensor name relative to the wrapped model, dim, offset, group id)
_PermOp = tuple[str, int, int, int]


def _check_no_shared_parameters(model: nn.Module, graph: ChannelGraph) -> None:
    """Refuse tied weights on a resized layer: widening would silently untie them."""
    owners: dict[int, list[str]] = {}
    for mod_name, mod in model.named_modules(remove_duplicate=False):
        for p_name, p in mod.named_parameters(recurse=False):
            owners.setdefault(id(p), []).append(f"{mod_name}.{p_name}" if mod_name else p_name)
    for names in owners.values():
        if len(names) > 1 and any(n.rsplit(".", 1)[0] in graph.layers for n in names):
            raise ValueError("shared (tied) parameters are not supported on layers NetAug resizes: " + ", ".join(names))


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


@torch.no_grad()
def _add_depth(work: nn.Module, example_inputs: Any, depth: Mapping[str, int]) -> tuple[list[str], list[str]]:
    """Replace each chosen block of ``work`` by a :class:`DepthStack`.

    Returns ``(depth_paths, zeroed_bn_paths)`` where ``zeroed_bn_paths`` lists every BN
    that :func:`make_extra` zero-initialized so widening keeps all its channels at zero.
    """
    paths: dict[str, int] = {}
    for path, k in depth.items():
        try:
            mod = work.get_submodule(path) if path else work
        except AttributeError as e:
            raise ValueError(f"depth: no module named {path!r}") from e
        if type(mod) is nn.Sequential and len(mod) > 0:  # a stage container: its last block
            path = f"{path}.{len(mod) - 1}" if path else str(len(mod) - 1)
            mod = work.get_submodule(path)
        if not path:
            raise ValueError("depth: the whole model cannot be repeated")
        if next(mod.parameters(), None) is None:
            raise ValueError(f"depth: {path!r} ({type(mod).__name__}) has no parameters to repeat")
        if int(k) < 1:
            raise ValueError(f"depth: {path!r} needs at least 1 extra block, got {k}")
        if path in paths:
            raise ValueError(f"depth: {path!r} is given twice")
        paths[path] = int(k)
    for a in paths:
        for b in paths:
            if a != b and b.startswith(a + "."):
                raise ValueError(f"depth: {b!r} is inside {a!r}")

    # record the input and output of every chosen block on the example input
    seen: dict[str, list[tuple[Any, Any]]] = {p: [] for p in paths}
    hooks = [
        work.get_submodule(p).register_forward_hook(lambda m, args, out, p=p: seen[p].append((args, out)))
        for p in paths
    ]
    flags = [(m, m.training) for m in work.modules()]  # keep per-module flags (frozen BN, ...)
    work.eval()
    try:
        work(*(example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)))
    finally:
        for h in hooks:
            h.remove()
        for m, flag in flags:
            m.training = flag
    zeroed_bns: list[str] = []
    for p, k in paths.items():
        calls = seen[p]
        if len(calls) != 1:
            raise ValueError(f"depth: {p!r} must run exactly once per forward (ran {len(calls)} times)")
        args, out = calls[0]
        if len(args) != 1 or not isinstance(args[0], torch.Tensor) or not isinstance(out, torch.Tensor):
            raise ValueError(f"depth: {p!r} must take one tensor and return one tensor")
        if args[0].shape != out.shape:
            raise ValueError(
                f"depth: {p!r} changes the shape {tuple(args[0].shape)} -> {tuple(out.shape)}; "
                "only shape-preserving blocks can be repeated"
            )
        block = work.get_submodule(p)
        extras = []
        for i in range(k):
            extra, zero_rel = make_extra(block, out)
            extras.append(extra)
            if zero_rel is not None:
                zeroed_bns.append(f"{p}.extra.{i}.{zero_rel}" if zero_rel else f"{p}.extra.{i}")
        stack = DepthStack(block, extras)
        stack.training = block.training  # only the wrapper: keep the flags inside the block
        _set_submodule(work, p, stack)
    return sorted(paths), zeroed_bns


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
        graph: a pre-built :class:`ChannelGraph` (skips ``torch.fx`` analysis). Not allowed
            together with ``depth``.
        update_bn_in_aug: if False (default), batch norm running statistics are only updated
            in base mode, so augmented passes cannot pollute the base model's statistics.
        depth: depth augmentation, ``{module path: number of extra blocks}``. The augmented
            models may run up to that many extra copies of the module right after it. The
            module must keep the shape of its input and have parameters. A path to a plain
            ``nn.Sequential`` container (a ResNet stage such as ``"layer1"``; not a subclass)
            means its last block. See :mod:`torchnetaug.depth`.

    Modes:
        * ``set_base()``: the original tiny model (used for evaluation and export).
        * ``set_max()``: every free group at its widest, every extra block on.
        * ``set_random()``: each free group samples a width from its choices, and each
          depth point samples how many extra blocks run.
    """

    _template: nn.Module

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Any = None,
        width_mults: Sequence[float] = (1.0, 2.0, 3.0),
        graph: ChannelGraph | None = None,
        update_bn_in_aug: bool = False,
        depth: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        check_plain(model)
        mults = sorted(float(m) for m in width_mults)
        if not mults or mults[0] != 1.0:
            raise ValueError("width_mults must contain 1.0 as its smallest value")
        work = copy.deepcopy(model)
        self._depth_paths: list[str] = []
        zeroed_bns: list[str] = []
        if depth:
            if graph is not None:
                raise ValueError("depth cannot be combined with a pre-built graph")
            if example_inputs is None:
                raise ValueError("example_inputs is required for depth augmentation")
            self._depth_paths, zeroed_bns = _add_depth(work, example_inputs, depth)
        if graph is None:
            if example_inputs is None:
                raise ValueError("example_inputs is required when graph is not given")
            graph = analyze(work, example_inputs)
        _check_no_shared_parameters(work, graph)
        self.graph = graph
        self.width_mults = tuple(mults)
        self.update_bn_in_aug = update_bn_in_aug

        self.base_sizes: list[int] = list(graph.group_sizes)
        self.choices: list[list[int]] = []
        for size, fixed in zip(graph.group_sizes, graph.group_fixed):
            if fixed:
                self.choices.append([size])
            else:
                self.choices.append(sorted({max(1, round(size * m)) for m in mults}))
        self.max_sizes: list[int] = [c[-1] for c in self.choices]

        # Plain copy of the original model: export() loads sliced weights into a copy of it.
        self.__dict__["_template"] = copy.deepcopy(model)

        self.model = work
        self.max_depths: dict[str, int] = {p: len(self._stack(p).extra) for p in self._depth_paths}
        self.base_depths: dict[str, int] = {p: 0 for p in self._depth_paths}
        self.active_depths: dict[str, int] = dict(self.base_depths)
        self._in_offsets: dict[str, list[int]] = {}
        for name, role in graph.layers.items():
            offsets, off = [], 0
            for g in role.in_groups:
                offsets.append(off)
                off += self.max_sizes[g]
            self._in_offsets[name] = offsets
            old = self.model.get_submodule(name)
            _set_submodule(self.model, name, self._widen(name, old))
        # Keep residual extra blocks as the identity across *every* widened channel, not just base.
        with torch.no_grad():
            for bn_path in zeroed_bns:
                bn = self.model.get_submodule(bn_path)
                if isinstance(bn.weight, torch.Tensor):
                    bn.weight.zero_()
                if isinstance(bn.bias, torch.Tensor):
                    bn.bias.zero_()

        self.mode = "base"
        self.active_sizes: list[int] = list(self.base_sizes)
        self.set_base()

    # ------------------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------------------
    def _base_in_index(self, name: str, device: torch.device | None) -> torch.Tensor:
        role = self.graph.layers[name]
        segs = [(o, self.base_sizes[g]) for o, g in zip(self._in_offsets[name], role.in_groups)]
        return segments_index(tuple(segs), device)

    @torch.no_grad()
    def _widen(self, name: str, old: nn.Module) -> nn.Module:
        role = self.graph.layers[name]
        in_max = sum(self.max_sizes[g] for g in role.in_groups)
        p = next(old.parameters(), None)
        if p is None:
            p = next((b for b in old.buffers() if b.is_floating_point()), None)
        fk: dict[str, Any] = {"device": p.device, "dtype": p.dtype} if p is not None else {}
        idx = self._base_in_index(name, p.device if p is not None else None)

        if isinstance(old, nn.Conv2d):
            common = dict(
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                dilation=old.dilation,
                bias=old.bias is not None,
                padding_mode=old.padding_mode,
                **fk,
            )
            if role.kind == "dense":
                assert role.out_group is not None
                out_max, out_base = self.max_sizes[role.out_group], self.base_sizes[role.out_group]
                new = DynamicConv2d(in_max, out_max, groups=1, **common)
                new.weight[:out_base].index_copy_(1, idx, old.weight)
                if old.bias is not None and new.bias is not None:
                    new.bias[:out_base].copy_(old.bias)
            else:
                new = DynamicConv2d(in_max, in_max, groups=in_max, **common)
                new.weight.index_copy_(0, idx, old.weight)
                if old.bias is not None and new.bias is not None:
                    new.bias.index_copy_(0, idx, old.bias)
        elif isinstance(old, nn.Linear):
            assert role.out_group is not None
            out_max, out_base = self.max_sizes[role.out_group], self.base_sizes[role.out_group]
            new = DynamicLinear(in_max, out_max, bias=old.bias is not None, **fk)
            new.weight[:out_base].index_copy_(1, idx, old.weight)
            if old.bias is not None and new.bias is not None:
                new.bias[:out_base].copy_(old.bias)
        elif isinstance(old, nn.modules.batchnorm._BatchNorm):
            new = DynamicBatchNorm(
                in_max,
                eps=old.eps,
                momentum=old.momentum,
                affine=old.affine,
                track_running_stats=old.track_running_stats,
                **fk,
            )
            for attr in ("weight", "bias", "running_mean", "running_var"):
                src = getattr(old, attr)
                if src is not None:
                    getattr(new, attr).index_copy_(0, idx, src)
            if old.track_running_stats and new.num_batches_tracked is not None and old.num_batches_tracked is not None:
                new.num_batches_tracked.copy_(old.num_batches_tracked)
        else:  # pragma: no cover - the analysis only emits the types above
            raise TypeError(f"cannot widen {type(old).__name__}")
        for (n_old, p_old), (n_new, p_new) in zip(old.named_parameters(), new.named_parameters()):
            p_new.requires_grad_(p_old.requires_grad)
        new.train(old.training)
        return new

    def dynamic_layers(self) -> Iterator[tuple[str, DynamicModule]]:
        for name in self.graph.layers:
            yield name, cast(DynamicModule, self.model.get_submodule(name))

    def num_active_params(self) -> int:
        """Number of parameters the current width uses.

        This equals the parameter count of a plain model built at the current sizes
        (for the base sizes: the parameter count of ``export()``).
        """
        dyn = dict(self.dynamic_layers())
        off = [f"{p}.extra.{i}" for p, d in self.active_depths.items() for i in range(d, self.max_depths[p])]
        total = 0
        for name, mod in self.model.named_modules():
            if any(name == o or name.startswith(o + ".") for o in off):
                continue  # extra block that does not run
            if name in dyn:
                if isinstance(mod, (DynamicConv2d, DynamicLinear)):
                    total += mod.active_weight().numel()
                    b = mod.active_bias()
                    total += 0 if b is None else b.numel()
                elif mod.affine:  # batch norm: one weight and one bias per channel
                    total += 2 * mod.active_in
            elif not any(name.startswith(d + ".") for d in dyn):
                total += sum(p.numel() for p in mod.parameters(recurse=False))
        return total

    # ------------------------------------------------------------------------------
    # width control
    # ------------------------------------------------------------------------------
    def _stack(self, path: str) -> DepthStack:
        return cast(DepthStack, self.model.get_submodule(path))

    @property
    def is_base(self) -> bool:
        """True when the active network is the original tiny model."""
        return self.active_sizes == self.base_sizes and self.active_depths == self.base_depths

    def state(self) -> tuple[list[int], dict[str, int], str]:
        """The active widths, depths and mode; :meth:`set_state` restores them."""
        return list(self.active_sizes), dict(self.active_depths), self.mode

    def set_state(self, state: tuple[Sequence[int], Mapping[str, int], str]) -> None:
        sizes, depths, mode = state
        self._apply_state(sizes, depths, mode)

    def _apply_state(self, sizes: Sequence[int], depths: Mapping[str, int], mode: str) -> None:
        if len(sizes) != self.graph.num_groups:
            raise ValueError(f"expected {self.graph.num_groups} sizes, got {len(sizes)}")
        for g, s in enumerate(sizes):
            if not 1 <= s <= self.max_sizes[g]:
                raise ValueError(f"group {g}: size {s} outside [1, {self.max_sizes[g]}]")
            if self.graph.group_fixed[g] and s != self.base_sizes[g]:
                raise ValueError(f"group {g} is fixed at {self.base_sizes[g]} channels")
        if set(depths) != set(self.max_depths):
            raise ValueError(f"expected depths for {sorted(self.max_depths)}, got {sorted(depths)}")
        for p, d in depths.items():
            if not 0 <= d <= self.max_depths[p]:
                raise ValueError(f"depth {p!r}: {d} extra blocks outside [0, {self.max_depths[p]}]")
        self.active_sizes = [int(s) for s in sizes]
        self.active_depths = {p: int(d) for p, d in depths.items()}
        self.mode = mode
        for p, d in self.active_depths.items():
            self._stack(p).active = d
        update = self.is_base or self.update_bn_in_aug
        for name, layer in self.dynamic_layers():
            role = self.graph.layers[name]
            layer.set_in_segments([(o, self.active_sizes[g]) for o, g in zip(self._in_offsets[name], role.in_groups)])
            if role.kind == "dense" and role.out_group is not None:
                layer.active_out = self.active_sizes[role.out_group]
            if isinstance(layer, DynamicBatchNorm):
                layer.update_stats = update

    def set_sizes(self, sizes: Sequence[int], mode: str = "custom") -> None:
        """Set the active width of every channel group (one size per group). Depth is kept."""
        self._apply_state(sizes, self.active_depths, mode)

    def set_depths(self, depths: Mapping[str, int], mode: str = "custom") -> None:
        """Set how many extra blocks run at each depth point (0 = base). Width is kept.

        Depth points not in ``depths`` keep their current value.
        """
        self._apply_state(self.active_sizes, {**self.active_depths, **depths}, mode)

    def set_base(self) -> None:
        self._apply_state(self.base_sizes, self.base_depths, "base")

    def set_max(self) -> None:
        self._apply_state(self.max_sizes, self.max_depths, "max")

    def set_width(self, mult: float) -> None:
        """Scale every free group by the same multiplier (``mode == "custom"``). Depth is kept.

        ``mult`` may lie between the training choices (e.g. 1.5), which is useful to check
        how accuracy changes with width. Sizes are clamped to ``[1, max_size]``.
        """
        sizes = [
            b if fixed else min(mx, max(1, round(b * mult)))
            for b, mx, fixed in zip(self.base_sizes, self.max_sizes, self.graph.group_fixed)
        ]
        base = sizes == self.base_sizes and self.active_depths == self.base_depths
        self._apply_state(sizes, self.active_depths, "base" if base else "custom")

    def set_random(self, generator: random.Random | None = None, granularity: str = "group") -> None:
        """Sample an augmented model.

        ``granularity="group"`` samples each free group on its own (a ResNet stage, an
        inverted-block expansion, ...). ``"global"`` samples one multiplier for the whole
        model. Each depth point samples its number of extra blocks uniformly from
        ``0 .. max``. Pass a seeded ``random.Random`` to make sampling reproducible or to
        keep several distributed workers in sync.
        """
        rng = generator or random
        if granularity == "group":
            sizes = [rng.choice(c) for c in self.choices]
        elif granularity == "global":
            m = rng.choice(self.width_mults)
            sizes = [b if fixed else max(1, round(b * m)) for b, fixed in zip(self.base_sizes, self.graph.group_fixed)]
        else:
            raise ValueError("granularity must be 'group' or 'global'")
        depths = {p: rng.randint(0, k) for p, k in self.max_depths.items()}
        self._apply_state(sizes, depths, "random")

    @contextmanager
    def mode_ctx(self, mode: str, **kwargs: Any) -> Generator[NetAugModel, None, None]:
        """Temporarily switch mode (``"base"``, ``"max"`` or ``"random"``)."""
        previous = self.state()
        getattr(self, f"set_{mode}")(**kwargs)
        try:
            yield self
        finally:
            self.set_state(previous)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------------------
    @torch.no_grad()
    def export(self, sizes: Sequence[int] | None = None, depths: Mapping[str, int] | None = None) -> nn.Module:
        """Return a plain PyTorch model holding the weights of one width and depth.

        Without arguments (or with the base sizes and depths) the result is a copy of the
        original model, same class and same shapes. With other sizes (for example
        ``model.export(model.active_sizes)`` after ``set_width(2.0)``) the result keeps the
        original model's structure and ``forward``, with every resized layer replaced by a
        plain ``nn.Conv2d`` / ``nn.Linear`` / batch-norm layer of the active size. With
        ``depths``, each depth point becomes ``nn.Sequential(block, *extra_blocks)``.
        Neither result depends on this package.
        """
        sizes = self.base_sizes if sizes is None else [int(s) for s in sizes]
        depths = self.base_depths if depths is None else {**self.base_depths, **depths}
        if sizes != self.base_sizes or depths != self.base_depths:
            return self._export_state(sizes, depths)
        with self.mode_ctx("base"):
            state = self.model.state_dict()
            for name, layer in self.dynamic_layers():
                prefix = name + "."
                if isinstance(layer, (DynamicConv2d, DynamicLinear)):
                    state[prefix + "weight"] = layer.active_weight()
                    b = layer.active_bias()
                    if b is not None:
                        state[prefix + "bias"] = b
                else:
                    for attr in ("weight", "bias", "running_mean", "running_var"):
                        if getattr(layer, attr) is not None:
                            state[prefix + attr] = layer._select_in(getattr(layer, attr), 0)
            out = copy.deepcopy(self._template)
            cloned = OrderedDict()
            for k, v in state.items():
                k = self._template_name(k)
                if k is not None:
                    cloned[k] = v.detach().clone()
            # keep the per-module version info: some modules (e.g. MnasNet) require it
            meta = getattr(state, "_metadata", None)
            if meta is not None:
                meta = OrderedDict(
                    (self._template_name(k) if k else k, v)
                    for k, v in meta.items()
                    if not k or self._template_name(k) is not None
                )
            cloned.__dict__["_metadata"] = meta
            out.load_state_dict(cloned, assign=True)  # keep the model's dtype (e.g. after .double())
            # buffers outside the state_dict (persistent=False) follow the model too
            model_buffers = {self._template_name(k): v for k, v in self.model.named_buffers(remove_duplicate=False)}
            copies: dict[int, torch.Tensor] = {}  # one copy per model tensor: ties follow the model
            for name, buf in list(out.named_buffers(remove_duplicate=False)):
                src = model_buffers.get(name)
                if name in cloned or src is None or not buf.is_floating_point():
                    continue
                if id(src) not in copies:
                    copies[id(src)] = (
                        src.detach().clone() if src.shape == buf.shape else buf.to(device=src.device, dtype=src.dtype)
                    )
                owner_name, _, attr = name.rpartition(".")
                owner = out.get_submodule(owner_name) if owner_name else out
                owner._buffers[attr] = copies[id(src)]
        p = next(self.model.parameters(), None)
        if p is not None:
            out.to(p.device)
        # preserve per-module requires_grad and training flags (e.g. frozen BN / weights)
        model_params = {
            self._template_name(k): param for k, param in self.model.named_parameters(remove_duplicate=False)
        }
        for name, param in out.named_parameters(remove_duplicate=False):
            src = model_params.get(name)
            if src is not None:
                param.requires_grad_(src.requires_grad)
        out.train(self.training)
        for name, mod in self.model.named_modules(remove_duplicate=False):
            tname = self._template_name(name)
            if tname is not None:
                out.get_submodule(tname).training = mod.training
        return out

    def _template_name(self, name: str) -> str | None:
        """Name in the original model of a tensor or module of ``self.model``.

        For a depth point ``P``: ``P.block.x`` -> ``P.x``, ``P.block`` -> ``P``, and names
        under ``P.extra`` (blocks the original model does not have) -> None.
        """
        for p in self._depth_paths:
            if name == p + ".block":
                return p
            if name.startswith(p + ".block."):
                return p + "." + name[len(p) + len(".block.") :]
            if name == p or name == p + ".extra" or name.startswith(p + ".extra."):
                return None
        return name

    def _template_layer_name(self, name: str) -> str:
        """Name in the original model of the layer that ``name`` is (or copies)."""
        for p in self._depth_paths:
            if name == p + ".block" or name.startswith(p + ".block."):
                return p + name[len(p + ".block") :]
            if name.startswith(p + ".extra."):
                _, _, rest = name[len(p + ".extra.") :].partition(".")  # drop the copy index
                return p + ("." + rest if rest else "")
        return name

    def _export_state(self, sizes: Sequence[int], depths: Mapping[str, int]) -> nn.Module:
        previous = self.state()
        self._apply_state(sizes, depths, "custom")
        try:
            out = copy.deepcopy(self.model)
            for name, layer in self.dynamic_layers():
                _set_submodule(out, name, self._plain_layer(name, layer))
            for p in self._depth_paths:
                _set_submodule(out, p, cast(DepthStack, out.get_submodule(p)).to_plain())
        finally:
            self.set_state(previous)
        return out

    def _plain_layer(self, name: str, layer: DynamicModule) -> nn.Module:
        """A plain layer of the original layer's class holding ``layer``'s active slice."""
        cls: Any = type(self._template.get_submodule(self._template_layer_name(name)))
        n_in = layer.active_in
        tensors = list(layer.parameters()) + list(layer.buffers())
        device = tensors[0].device if tensors else torch.device("cpu")
        fp = next((t for t in tensors if t.is_floating_point()), None)
        dtype = fp.dtype if fp is not None else None
        fk: dict[str, Any] = {"device": device, "dtype": dtype}
        with torch.no_grad():
            if isinstance(layer, DynamicConv2d):
                n_out = n_in if layer.depthwise else layer.active_out
                new: Any = cls(
                    n_in,
                    n_out,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    layer.dilation,
                    groups=n_in if layer.depthwise else 1,
                    bias=layer.bias is not None,
                    padding_mode=layer.padding_mode,
                    **fk,
                )
                new.weight.copy_(layer.active_weight())
                b = layer.active_bias()
                if b is not None:
                    new.bias.copy_(b)
            elif isinstance(layer, DynamicLinear):
                new = cls(n_in, layer.active_out, bias=layer.bias is not None, **fk)
                new.weight.copy_(layer.active_weight())
                b = layer.active_bias()
                if b is not None:
                    new.bias.copy_(b)
            else:
                assert isinstance(layer, DynamicBatchNorm)
                new = cls(
                    n_in,
                    eps=layer.eps,
                    momentum=layer.momentum,
                    affine=layer.affine,
                    track_running_stats=layer.track_running_stats,
                    **fk,
                )
                for attr in ("running_mean", "running_var"):
                    if getattr(layer, attr) is not None:
                        getattr(new, attr).copy_(layer._select_in(getattr(layer, attr), 0))
                if layer.num_batches_tracked is not None:
                    new.num_batches_tracked.copy_(layer.num_batches_tracked)
                if layer.affine and layer.weight is not None and layer.bias is not None:
                    new.weight.copy_(layer._select_in(layer.weight, 0))
                    new.bias.copy_(layer._select_in(layer.bias, 0))
        for (_, p_old), (_, p_new) in zip(layer.named_parameters(), new.named_parameters()):
            p_new.requires_grad_(p_old.requires_grad)
        new.train(layer.training)
        return new

    # ------------------------------------------------------------------------------
    # channel sorting
    # ------------------------------------------------------------------------------
    def _perm_ops(self) -> dict[int, list[_PermOp]]:
        """For each free group, list every tensor slice that carries its channels."""
        ops: dict[int, list[_PermOp]] = {g: [] for g in self.graph.free_groups()}
        for name, layer in self.dynamic_layers():
            role = self.graph.layers[name]
            if role.kind == "dense":
                if role.out_group is not None and role.out_group in ops:
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
    def channel_importance(self) -> dict[int, torch.Tensor]:
        """L1 importance of each channel of each free group (max-width layout).

        A channel is important if the layers that *read* it give it large weights. If a
        group has no dense reader, the producers' output weights are used instead.
        """
        tensors = dict(self.model.named_parameters())
        result: dict[int, torch.Tensor] = {}
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
        optimizer: torch.optim.Optimizer | None = None,
        extra_models: Iterable[NetAugModel] = (),
    ) -> dict[int, torch.Tensor]:
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
        self, perms: dict[int, torch.Tensor], optimizer: torch.optim.Optimizer | None = None
    ) -> None:
        """Apply ``{group_id: permutation}``: new channel ``i`` = old channel ``perm[i]``."""
        tensors: dict[str, torch.Tensor] = dict(self.model.named_parameters())
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
                    targets += [v for v in state.values() if torch.is_tensor(v) and v.shape == params[name].shape]
                for t in targets:
                    idx = torch.arange(off, off + perm.numel(), device=t.device)
                    src = t.index_select(dim, idx[perm.to(t.device)])
                    t.index_copy_(dim, idx, src)
