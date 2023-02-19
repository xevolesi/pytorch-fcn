from copy import deepcopy
from itertools import product

import pytest
import torch
from torchvision.models import VGG16_Weights, vgg16

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
    "n_classes, spatial_size, transplant_score_layer, init_upsampling_as_bilinear",
    list(product((1, 21), (224, 256), (True, False), (True, False))),
)
def test_fcn32vgg16(
    n_classes,
    spatial_size,
    transplant_score_layer,
    init_upsampling_as_bilinear,
    get_test_config,
):
    config = deepcopy(get_test_config)
    config.training.image_size = spatial_size
    config.model.n_classes = n_classes
    config.model.transplant_score_layer = transplant_score_layer
    config.model.init_upsampling_as_bilinear = init_upsampling_as_bilinear

    model = FCN32VGG16(config)
    input = torch.randn((1, 3, config.training.image_size, config.training.image_size))

    # We don't need gradients during this test.
    with torch.no_grad():
        output = model(input)
    assert tuple(output.shape) == (1, n_classes, *(config.training.image_size,) * 2)

    if init_upsampling_as_bilinear:
        upsampling_kernel = generate_bilinear_kernel(
            in_channels=model.up.in_channels,
            out_channels=model.up.out_channels,
            kernel_size=model.up.kernel_size[0],
            dtype=model.up.weight.dtype,
        )
        assert torch.allclose(upsampling_kernel, model.up.weight)
        assert model.up.bias is None

    if transplant_score_layer:
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        trans_w = (
            vgg.classifier[-1]
            .weight.data[: config.model.n_classes]
            .view(model.scorer.weight.shape)
        )
        assert torch.allclose(model.scorer.weight.data, trans_w)
