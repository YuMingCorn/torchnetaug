"""torchnetaug: Network Augmentation (NetAug) for tiny neural networks in plain PyTorch.

Reference: Han Cai, Chuang Gan, Ji Lin, Song Han. "Network Augmentation for Tiny Deep
Learning". ICLR 2022. https://arxiv.org/abs/2110.08890
"""

from .graph import ChannelGraph, LayerRole, TraceError, analyze
from .layers import DynamicBatchNorm, DynamicConv2d, DynamicLinear, DynamicModule
from .netaug import NetAugModel
from .train import netaug_step, recalibrate_bn

__all__ = [
    "ChannelGraph",
    "LayerRole",
    "TraceError",
    "analyze",
    "DynamicBatchNorm",
    "DynamicConv2d",
    "DynamicLinear",
    "DynamicModule",
    "NetAugModel",
    "netaug_step",
    "recalibrate_bn",
]

__version__ = "0.1.0"
