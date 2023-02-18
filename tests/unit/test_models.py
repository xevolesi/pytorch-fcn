from copy import deepcopy

import pytest
import torch

from source.models import FCN32VGG16
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


@pytest.mark.parametrize(
    "n_classes, spatial_size",
    [
        (1, 224),
        (12, 448),
        (21, 256),
        (19, 384),
    ],
)
def test_fcn32vgg16(n_classes, spatial_size, get_test_config):
    config = deepcopy(get_test_config)
    config.model.n_classes = n_classes
    config.training.image_size = spatial_size
    model = FCN32VGG16(config)
    input = torch.randn((1, 3, config.training.image_size, config.training.image_size))

    # We don't need gradients during this test.
    with torch.no_grad():
        output = model(input)
    assert tuple(output.shape) == (1, n_classes, *(config.training.image_size,) * 2)
