import pytest
import torch

from source.models.utils import generate_bilinear_kernel, linear2conv2d


@pytest.mark.parametrize(
    "module, input_spatial_feature_size",
    [
        (torch.nn.Linear(in_features=25088, out_features=4096, bias=True), (7, 7)),
        (torch.nn.Linear(4096, 4096), (1, 1)),
    ],
)
def test_linear2conv2d(module, input_spatial_feature_size):
    conv = linear2conv2d(module, input_spatial_feature_size)
    linear_w = module.weight
    linear_b = module.bias
    conv_w = conv.weight
    conv_b = conv.bias
    assert torch.allclose(conv_w.view(linear_w.shape), linear_w)
    assert torch.allclose(conv_b.view(linear_b.shape), linear_b)


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size", [(64, 128, 3), (128, 256, 3), (21, 21, 64)]
)
def test_generate_bilinear_kernel(in_channels, out_channels, kernel_size):
    conv_transpose = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, bias=False
    )
    bilinear_kernel = generate_bilinear_kernel(
        in_channels, out_channels, kernel_size, dtype=conv_transpose.weight.dtype
    )
    conv_transpose.weight.data.copy_(bilinear_kernel)
    assert torch.allclose(bilinear_kernel, conv_transpose.weight.data)
