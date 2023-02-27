from itertools import product

import pytest
import torch

from source.modules import _get_upsampling_weight, conv_layer, upsampling_layer


@pytest.mark.parametrize("bias, zero_init", list(product([True, False], [True, False])))
def test_conv_layer(bias, zero_init):
    random_tensor = torch.randn((1, 3, 5, 5)).to("cpu")
    conv = conv_layer(3, 9, 3, bias=bias, zero_init=zero_init).to("cpu")
    with torch.no_grad():
        _ = conv(random_tensor)
    for param in conv.parameters():
        if zero_init:
            assert torch.count_nonzero(param).item() == 0
    if bias:
        assert conv.bias is not None
    else:
        assert conv.bias is None


@pytest.mark.parametrize(
    "bias, bilinear, trainable",
    list(product([True, False], [True, False], [True, False])),
)
def test_upsampling_layer(bias, bilinear, trainable):
    random_tensor = torch.randn((1, 3, 5, 5)).to("cpu")
    up = upsampling_layer(3, 3, 64, 32, bias, bilinear, trainable).to("cpu")
    with torch.no_grad():
        _ = up(random_tensor)
    for param in up.parameters():
        assert param.requires_grad == trainable
    if bias:
        assert up.bias is not None
    else:
        assert up.bias is None
    if bilinear:
        assert torch.allclose(up.weight.data, _get_upsampling_weight(3, 3, 64))
    try:
        up = upsampling_layer(3, 4, 64, 32, bias, bilinear, trainable).to("cpu")
    except ValueError:
        assert True
