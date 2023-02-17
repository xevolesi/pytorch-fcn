import typing as ty

import torch
import numpy as np


def linear2conv2d(
    module: torch.nn.Module,
    input_spatial_feature_size: ty.Tuple[int, int],
) -> torch.nn.Conv2d:
    out_channels, in_channels = module.weight.shape
    if (
        in_channels % input_spatial_feature_size[0] != 0 
        or in_channels % input_spatial_feature_size[1] != 0
    ):
        raise ValueError(
            "Expected module's input channels divisible by spatial feature size"
        )
    in_channels /= input_spatial_feature_size[0] * input_spatial_feature_size[1]
    conv = torch.nn.Conv2d(
        in_channels=int(in_channels),
        out_channels=out_channels,
        kernel_size=input_spatial_feature_size,
    )
    conv.weight.data.copy_(module.weight.data.view(conv.weight.shape))
    conv.bias.data.copy_(module.bias.data.view(conv.bias.shape))
    return conv


def generate_bilinear_kernel(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:

    def expand_and_tile(arr: np.ndarray, tile_size: int) -> np.ndarray:
        return np.tile(np.expand_dims(arr, axis=-1), tile_size)

    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = expand_and_tile(
        expand_and_tile(filt, out_channels),
        in_channels,
    ).transpose(3, 2, 0, 1)
    return torch.as_tensor(filt, dtype=dtype)
