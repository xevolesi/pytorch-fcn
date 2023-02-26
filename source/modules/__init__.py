import numpy as np
import torch
from torch import nn


def conv_layer(
    in_channels: int,
    out_channles: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    bias: bool = True,
    zero_init: bool = True,
) -> nn.Conv2d:
    layer = nn.Conv2d(in_channels, out_channles, kernel_size, stride, padding, bias=bias)
    if zero_init:
        layer.weight.data.zero_()
        if bias:
            layer.bias.data.zero_()
    return layer


def _get_upsampling_weight(
    in_channels: int, out_channels: int, kernel_size: int
) -> torch.Tensor:
    """
    Make a 2D bilinear kernel suitable for unsampling
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (
        1 - abs(og[1] - center) / factor
    )
    weight = np.zeros(
        (in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32
    )
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight).float()


def upsampling_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    bias: bool = False,
    bilinear: bool = True,
    trainable: bool = False,
) -> nn.ConvTranspose2d:
    layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
    if bilinear:
        initial_weight = _get_upsampling_weight(in_channels, out_channels, kernel_size)
        layer.weight.data.copy_(initial_weight)
    if not trainable:
        layer.weight.requires_grad = False
    return layer
