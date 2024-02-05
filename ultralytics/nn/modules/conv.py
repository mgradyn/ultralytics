# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F


__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "MobileOneBlock",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class DepthWiseConv(nn.Module):
    def __init__(self, inc, kernel_size, stride=1):
        super().__init__()
        padding = 1
        if kernel_size == 1:
            padding = 0
        # self.conv = nn.Sequential(
        #     nn.Conv2d(inc, inc, kernel_size, stride, padding, groups=inc, bias=False,),
        #     nn.BatchNorm2d(inc),
        # )
        self.conv = conv_bn(inc, inc, kernel_size, stride, padding, inc)

    def forward(self, x):
        return self.conv(x)


class PointWiseConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(inc, outc, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(outc),
        # )
        self.conv = conv_bn(inc, outc, 1, 1, 0)

    def forward(self, x):
        return self.conv(x)
    
class SEBlock(nn.Module):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x

class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            # print(self.rbr_conv[ix])
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
