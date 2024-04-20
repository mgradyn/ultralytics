# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPELAN,
    SPPF,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fGhost,
    C2fGhost_2,
    C2fMobilenet,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    GhostBottleneck,
    GhostBottleneck_2,
    HGBlock,
    HGStem,
    MobileOne,
    Proto,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    Silence,
    SPPFGhost,
    SPPFMobilenet,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvBnHswish,
    ConvTranspose,
    DepthwiseSeparableConv,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    ConvBnHswish,
    DepthwiseSeparableConv,
    LightConv,
    MobileOneBlock,
    RepConv,
    SpatialAttention,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ConvBnHswish",
    "DepthwiseSeparableConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "SPPFGhost",
    "SPPFMobilenet",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fGhost",
    "C2fGhost_2",
    "C2fMobilenet",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "GhostBottleneck_2",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
)
