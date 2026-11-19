"""Channel-group analysis.

A *channel group* is a set of channels that must always have the same width and the same
channel order. Examples:

* the output of a dense conv and the input of every layer that reads it;
* both sides of a residual ``x + f(x)`` (so every block in a ResNet stage shares one group);
* the input and output of a depthwise conv or a batch norm.

``torch.cat`` along the channel dimension does not merge groups. Instead, the result is a
*segment list*: one segment per concatenated group, in order.

The analysis uses a whitelist. Only operations listed here let channels pass through.
Every other operation *fixes* the channel groups it touches, so they keep their original
width and order. This is conservative: an unknown op can make the augmentation smaller,
but it can never make the model wrong.

You can also build a :class:`ChannelGraph` by hand for models that ``torch.fx`` cannot trace.
"""

from __future__ import annotations

import copy
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata

__all__ = ["LayerRole", "ChannelGraph", "analyze", "TraceError"]


class TraceError(RuntimeError):
    """Raised when ``torch.fx`` cannot trace the model."""


@dataclass(frozen=True)
class LayerRole:
    """How one layer touches channel groups.

    ``kind`` is one of:

    * ``"dense"``: conv (groups=1) or linear. Its output channels form ``out_group``;
      its input channels follow ``in_groups``.
    * ``"member"``: batch norm or depthwise conv. Its channels (input == output)
      follow ``in_groups``.
    """

    kind: str
    in_groups: Tuple[int, ...]
    out_group: Optional[int] = None


@dataclass
class ChannelGraph:
    """Result of the analysis: group sizes, which groups are fixed, and each layer's role."""

    group_sizes: List[int]
    group_fixed: List[bool]
    layers: Dict[str, LayerRole]
    fixed_reasons: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def num_groups(self) -> int:
        return len(self.group_sizes)

    def free_groups(self) -> List[int]:
        return [g for g, fixed in enumerate(self.group_fixed) if not fixed]

    def summary(self) -> str:
        lines = [f"ChannelGraph: {self.num_groups} groups, {len(self.free_groups())} augmentable"]
        for g, (size, fixed) in enumerate(zip(self.group_sizes, self.group_fixed)):
            producers = [n for n, r in self.layers.items() if r.kind == "dense" and r.out_group == g]
            status = "fixed" if fixed else "free"
            reason = ""
            if fixed and self.fixed_reasons.get(g):
                reason = " (" + "; ".join(sorted(set(self.fixed_reasons[g]))[:3]) + ")"
            lines.append(f"  group {g}: {size} ch, {status}{reason}, producers={producers}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# whitelists
# --------------------------------------------------------------------------------------

_PASSTHROUGH_MODULES = (
    nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, nn.Hardswish, nn.Hardsigmoid, nn.Sigmoid,
    nn.Tanh, nn.LeakyReLU, nn.ELU, nn.Mish, nn.Identity, nn.Dropout, nn.Dropout1d,
    nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.MaxPool1d, nn.MaxPool2d,
    nn.AvgPool1d, nn.AvgPool2d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.Upsample, nn.UpsamplingNearest2d,
    nn.UpsamplingBilinear2d,
)

_PASSTHROUGH_FUNCTIONS = {
    F.relu, F.relu_, F.relu6, F.silu, F.gelu, F.hardswish, F.hardsigmoid, F.sigmoid,
    F.tanh, F.leaky_relu, F.elu, F.mish, F.dropout, F.dropout1d, F.dropout2d,
    F.dropout3d, F.max_pool1d, F.max_pool2d, F.avg_pool1d, F.avg_pool2d,
    F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_max_pool1d,
    F.adaptive_max_pool2d, F.interpolate, torch.relu, torch.relu_, torch.sigmoid,
    torch.tanh, torch.clamp, torch.clip,
}

try:  # stochastic depth drops whole samples or rows, never single channels
    from torchvision.ops import stochastic_depth as _stochastic_depth

    _PASSTHROUGH_FUNCTIONS.add(_stochastic_depth)
except ImportError:  # pragma: no cover - torchvision is optional
    pass

# Calls that only read metadata (x.size(1), x.shape, x.dim()): they never change channels.
_METADATA_METHODS = {"size", "dim", "numel"}

_PASSTHROUGH_METHODS = {
    "relu", "relu_", "sigmoid", "sigmoid_", "tanh", "tanh_", "contiguous", "clone",
    "clamp", "clamp_", "clip", "clip_", "detach",
}

_RESHAPE_FUNCTIONS = {torch.flatten, torch.reshape, torch.squeeze}
_RESHAPE_METHODS = {"flatten", "view", "reshape", "squeeze"}

_BINARY_FUNCTIONS = {
    operator.add, operator.iadd, operator.sub, operator.isub, operator.mul,
    operator.imul, operator.truediv, operator.itruediv, torch.add, torch.sub,
    torch.mul, torch.div,
}
_BINARY_METHODS = {"add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_"}

_CAT_FUNCTIONS = {torch.cat, torch.concat, torch.concatenate}
_MEAN_FUNCTIONS = {torch.mean, torch.amax, torch.sum}
_MEAN_METHODS = {"mean", "amax", "sum"}


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------

Spec = Tuple[int, ...]  # ordered group ids along the channel dimension (dim 1)


def _shape(node: Any) -> Optional[torch.Size]:
    if not isinstance(node, fx.Node):
        return None
    meta = node.meta.get("tensor_meta")
    if isinstance(meta, TensorMetadata):
        return meta.shape
    return None


class _Analyzer:
    def __init__(self, gm: fx.GraphModule) -> None:
        self.gm = gm
        self.modules = dict(gm.named_modules())
        self.parent: List[int] = []
        self.size: List[int] = []
        self.fixed: List[bool] = []
        self.reasons: Dict[int, List[str]] = {}
        self.spec: Dict[fx.Node, Spec] = {}
        self.layers: Dict[str, LayerRole] = {}
        self.call_count: Dict[str, int] = {}
        for node in gm.graph.nodes:
            if node.op == "call_module":
                self.call_count[node.target] = self.call_count.get(node.target, 0) + 1

    # ---- union-find --------------------------------------------------------------
    def new_group(self, size: int, fixed: bool = False, reason: str = "") -> int:
        g = len(self.parent)
        self.parent.append(g)
        self.size.append(int(size))
        self.fixed.append(fixed)
        if fixed:
            self.reasons[g] = [reason]
        return g

    def find(self, g: int) -> int:
        while self.parent[g] != g:
            self.parent[g] = self.parent[self.parent[g]]
            g = self.parent[g]
        return g

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] != self.size[rb]:
            raise RuntimeError(f"cannot merge channel groups of size {self.size[ra]} and {self.size[rb]}")
        self.parent[rb] = ra
        self.fixed[ra] = self.fixed[ra] or self.fixed[rb]
        self.reasons.setdefault(ra, []).extend(self.reasons.pop(rb, []))

    def fix(self, spec: Optional[Spec], reason: str) -> None:
        for g in spec or ():
            r = self.find(g)
            self.fixed[r] = True
            self.reasons.setdefault(r, []).append(reason)

    def spec_size(self, spec: Spec) -> int:
        return sum(self.size[self.find(g)] for g in spec)

    # ---- helpers -----------------------------------------------------------------
    def input_specs(self, node: fx.Node) -> List[Spec]:
        return [self.spec[n] for n in node.all_input_nodes if n in self.spec]

    def opaque(self, node: fx.Node, reason: str) -> None:
        """Unknown op: fix all channel groups it reads, give its output a new fixed group."""
        for s in self.input_specs(node):
            self.fix(s, reason)
        shape = _shape(node)
        if shape is not None and len(shape) >= 2:
            self.spec[node] = (self.new_group(shape[1], True, reason),)

    def passthrough(self, node: fx.Node, src: Any) -> None:
        shape, src_shape = _shape(node), _shape(src)
        if (
            isinstance(src, fx.Node) and src in self.spec and shape is not None
            and src_shape is not None and len(shape) >= 2 and len(src_shape) >= 2
            and shape[0] == src_shape[0] and shape[1] == src_shape[1]
        ):
            self.spec[node] = self.spec[src]
            for other in node.all_input_nodes:
                if other is not src and other in self.spec:
                    self.fix(self.spec[other], f"extra tensor input of {node.name}")
        else:
            self.opaque(node, f"channel count changed at {node.name}")

    def merge_specs(self, node: fx.Node, specs: Sequence[Spec]) -> Spec:
        first = specs[0]
        same = all(
            len(s) == len(first) and all(self.size[self.find(a)] == self.size[self.find(b)] for a, b in zip(s, first))
            for s in specs
        )
        if not same:
            for s in specs:
                self.fix(s, f"mismatched channel layout at {node.name}")
            return first
        for s in specs[1:]:
            for a, b in zip(first, s):
                self.union(a, b)
        return first

    # ---- handlers ----------------------------------------------------------------
    def run(self) -> None:
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                shape = _shape(node)
                if shape is not None and len(shape) >= 2:
                    self.spec[node] = (self.new_group(shape[1], True, "model input"),)
            elif node.op == "get_attr":
                pass  # parameters used directly: handled where they are consumed
            elif node.op == "call_module":
                self.call_module(node)
            elif node.op in ("call_function", "call_method"):
                self.call_function(node)
            elif node.op == "output":
                for s in self.input_specs(node):
                    self.fix(s, "model output")

    def call_module(self, node: fx.Node) -> None:
        name = node.target
        mod = self.modules[name]
        x = node.args[0] if node.args else None
        in_spec = self.spec.get(x) if isinstance(x, fx.Node) else None
        in_shape = _shape(x)

        if isinstance(mod, _PASSTHROUGH_MODULES):
            return self.passthrough(node, x)
        if isinstance(mod, nn.Flatten):
            return self.reshape_like(node, x)

        if not isinstance(mod, (nn.Conv2d, nn.Linear, nn.modules.batchnorm._BatchNorm)):
            return self.opaque(node, f"unsupported module {type(mod).__name__} ({name})")
        if self.call_count[name] > 1:
            return self.opaque(node, f"module {name} is called more than once")
        if in_spec is None or in_shape is None or len(node.args) != 1 or node.kwargs:
            return self.opaque(node, f"unexpected input to {name}")

        c_in = self.spec_size(in_spec)
        if isinstance(mod, nn.Conv2d):
            if type(mod) is not nn.Conv2d or c_in != mod.in_channels:
                return self.opaque(node, f"unsupported conv {name}")
            if mod.groups == 1:
                g = self.new_group(mod.out_channels)
                self.layers[name] = LayerRole("dense", in_spec, g)
                self.spec[node] = (g,)
            elif mod.groups == mod.in_channels == mod.out_channels:
                self.layers[name] = LayerRole("member", in_spec)
                self.spec[node] = in_spec
            else:
                self.opaque(node, f"grouped conv {name} (groups={mod.groups})")
        elif isinstance(mod, nn.Linear):
            if type(mod) is not nn.Linear or len(in_shape) != 2 or c_in != mod.in_features:
                return self.opaque(node, f"linear {name} on non-2D input")
            g = self.new_group(mod.out_features)
            self.layers[name] = LayerRole("dense", in_spec, g)
            self.spec[node] = (g,)
        else:  # batch norm
            if c_in != mod.num_features or not isinstance(
                mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
            ):
                return self.opaque(node, f"unsupported norm {name}")
            self.layers[name] = LayerRole("member", in_spec)
            self.spec[node] = in_spec

    def reshape_like(self, node: fx.Node, x: Any) -> None:
        """Keep the channel groups if dim 0 and dim 1 are unchanged (e.g. flatten after pooling)."""
        s, o = _shape(x), _shape(node)
        if (
            s is not None and o is not None and len(s) >= 2 and len(o) >= 2
            and s[0] == o[0] and s[1] == o[1] and math.prod(s[2:]) == math.prod(o[2:])
        ):
            return self.passthrough(node, x)
        self.opaque(node, f"reshape changes channels at {node.name}")

    def call_function(self, node: fx.Node) -> None:
        t = node.target
        is_method = node.op == "call_method"
        out_shape = _shape(node)

        if (not is_method and t in _BINARY_FUNCTIONS) or (is_method and t in _BINARY_METHODS):
            return self.binary(node)
        if (not is_method and t in _CAT_FUNCTIONS):
            return self.cat(node)
        if (not is_method and t in _PASSTHROUGH_FUNCTIONS) or (is_method and t in _PASSTHROUGH_METHODS):
            return self.passthrough(node, node.args[0] if node.args else None)
        if (not is_method and t in _RESHAPE_FUNCTIONS) or (is_method and t in _RESHAPE_METHODS):
            return self.reshape_like(node, node.args[0])
        if (not is_method and t in _MEAN_FUNCTIONS) or (is_method and t in _MEAN_METHODS):
            return self.reduce(node)
        if (is_method and t in _METADATA_METHODS) or (not is_method and t is getattr):
            if out_shape is None:
                return  # metadata read such as x.size(0) or x.shape
        if out_shape is None and not self.input_specs(node):
            return  # other non-tensor helper
        self.opaque(node, f"unsupported op {getattr(t, '__name__', t)} ({node.name})")

    def binary(self, node: fx.Node) -> None:
        out = _shape(node)
        if out is None or len(out) < 2:
            return self.opaque(node, f"non-channel binary op {node.name}")
        participants: List[Spec] = []
        for arg in list(node.args[:2]) + list(node.kwargs.values()):
            if not isinstance(arg, fx.Node):
                continue  # python scalar
            s = _shape(arg)
            if s is None:
                continue  # non-tensor value
            ch_dim = 1 - (len(out) - len(s))
            if ch_dim < 0 or s[ch_dim] == 1:
                # Broadcasts over channels. A 1-channel tensor (e.g. the map of a spatial
                # attention conv with out_channels=1) must keep exactly 1 channel.
                if ch_dim == 1 and out[1] != 1 and arg in self.spec:
                    self.fix(self.spec[arg], f"1-channel tensor broadcast in {node.name}")
                continue
            if ch_dim != 1 or arg not in self.spec:
                return self.opaque(node, f"channel-wise constant in {node.name}")
            participants.append(self.spec[arg])
        if not participants:
            return self.opaque(node, f"binary op without channel input {node.name}")
        self.spec[node] = self.merge_specs(node, participants)

    def cat(self, node: fx.Node) -> None:
        tensors = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        out = _shape(node)
        if out is None or not isinstance(tensors, (list, tuple)):
            return self.opaque(node, f"unsupported cat {node.name}")
        dim = dim % len(out)
        if not all(isinstance(a, fx.Node) and a in self.spec for a in tensors):
            return self.opaque(node, f"cat with unknown input {node.name}")
        specs = [self.spec[a] for a in tensors]
        if dim == 1:
            self.spec[node] = tuple(g for s in specs for g in s)
        elif dim >= 2:
            self.spec[node] = self.merge_specs(node, specs)
        else:
            self.opaque(node, f"cat along batch dim {node.name}")

    def reduce(self, node: fx.Node) -> None:
        x = node.args[0]
        s = _shape(x)
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        if s is None or dims is None:
            return self.opaque(node, f"full reduction {node.name}")
        dims = [dims] if isinstance(dims, int) else list(dims)
        dims = [d % len(s) for d in dims]
        if 0 in dims or 1 in dims:
            return self.opaque(node, f"reduction over channels {node.name}")
        self.passthrough(node, x)

    # ---- result ------------------------------------------------------------------
    def result(self) -> ChannelGraph:
        roots: Dict[int, int] = {}
        for g in range(len(self.parent)):
            r = self.find(g)
            if r not in roots:
                roots[r] = len(roots)

        def m(g: int) -> int:
            return roots[self.find(g)]

        sizes = [0] * len(roots)
        fixed = [False] * len(roots)
        reasons: Dict[int, List[str]] = {}
        for r, i in roots.items():
            sizes[i] = self.size[r]
            fixed[i] = self.fixed[r]
            if self.reasons.get(r):
                reasons[i] = list(self.reasons[r])
        layers = {
            name: LayerRole(role.kind, tuple(m(g) for g in role.in_groups),
                            None if role.out_group is None else m(role.out_group))
            for name, role in self.layers.items()
        }
        return ChannelGraph(sizes, fixed, layers, reasons)


def analyze(model: nn.Module, example_inputs: Any, tracer: Optional[fx.Tracer] = None) -> ChannelGraph:
    """Trace ``model`` with ``torch.fx`` and find its channel groups.

    ``example_inputs`` is a tensor or a tuple of tensors used to propagate shapes. The model
    is copied first, so its weights and batch-norm statistics are not changed.
    """
    model = copy.deepcopy(model).eval()
    try:
        if tracer is None:
            gm = fx.symbolic_trace(model)
        else:
            graph = tracer.trace(model)
            gm = fx.GraphModule(tracer.root, graph)
    except Exception as e:  # noqa: BLE001 - torch.fx raises many exception types
        raise TraceError(
            "torch.fx could not trace this model. Remove data-dependent control flow from "
            "forward(), pass a custom tracer, or build a torchnetaug.ChannelGraph by hand.\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    with torch.no_grad():
        ShapeProp(gm).propagate(*example_inputs)
    analyzer = _Analyzer(gm)
    analyzer.run()
    return analyzer.result()
