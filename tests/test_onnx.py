"""Exported models can be converted to ONNX and give the same output in onnxruntime.

Skipped when ``onnx``, ``onnxscript`` or ``onnxruntime`` is not installed (they are dev
dependencies, not package dependencies).
"""

from __future__ import annotations

import io
import random
import warnings
from typing import Any, cast

import pytest
import torch
import torchvision
from conftest import randomize
from test_widths import BASE_WIDTHS, TinyResNet

from torchnetaug import DynamicModule, NetAugModel

pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
ort = pytest.importorskip("onnxruntime")

MODELS = {
    "tiny_resnet": lambda: TinyResNet(BASE_WIDTHS),
    "mobilenet_v2": lambda: torchvision.models.mobilenet_v2(width_mult=0.35, num_classes=10),
}


def _to_onnx(model: torch.nn.Module, x: torch.Tensor, dynamo: bool) -> bytes:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dynamo:
            prog = torch.onnx.export(model, (x,), dynamo=True, verbose=False)
            assert prog is not None
            return prog.model_proto.SerializeToString()
        f = io.BytesIO()
        torch.onnx.export(model, (x,), cast(Any, f), dynamo=False, opset_version=17)
        return f.getvalue()


def _run(onnx_bytes: bytes, x: torch.Tensor) -> torch.Tensor:
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    return torch.from_numpy(sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0])


@pytest.mark.parametrize("dynamo", [True, False], ids=["dynamo", "torchscript"])
@pytest.mark.parametrize("width", ["base", "x2", "random"])
@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_exported_model_runs_in_onnxruntime(model_name, width, dynamo):
    torch.manual_seed(0)
    model = randomize(MODELS[model_name]()).float()
    x = torch.randn(2, 3, 32, 32, dtype=torch.float32)  # conftest sets float64 as default
    netaug = NetAugModel(model, x, width_mults=(1.0, 2.0, 3.0)).eval()
    if width == "base":
        exported = netaug.export()
    else:
        netaug.set_width(2.0) if width == "x2" else netaug.set_random(random.Random(1))
        exported = netaug.export(netaug.active_sizes)
    exported.eval()
    assert not any(isinstance(m, DynamicModule) for m in exported.modules())
    try:
        onnx_bytes = _to_onnx(exported, x, dynamo)
    except (AttributeError, NotImplementedError, RuntimeError) as e:
        if not dynamo and "dynamo" in str(e).lower():  # legacy exporter removed in this torch
            pytest.skip(f"TorchScript-based ONNX exporter unavailable: {e}")
        raise
    with torch.no_grad():
        expected = exported(x)
    torch.testing.assert_close(_run(onnx_bytes, x), expected, rtol=1e-4, atol=1e-5)
