from copy import deepcopy
from itertools import product

import pytest
import torch
from torchvision.models import vgg16

from source.models import FCN8s, FCN16s, FCN32s
from source.models.backbones import ConvolutionizedVGG16
from source.modules import _get_upsampling_weight


def _lazy_conv_params(model):
    for name, param in model.named_parameters():
        if "conv" in name:
            yield param


def _lazy_packed_conv_params(model):
    for module in filter(
        lambda module: isinstance(module, torch.nn.Conv2d), model.features
    ):
        yield from module.parameters()


def _check_upsamplings(model, bilinear, final_trainable, inter_trainable):
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ConvTranspose2d):
            continue

        # Check trainability.
        check_flag = final_trainable if name == "final_up" else inter_trainable
        for param in module.parameters():
            assert param.requires_grad == check_flag

        # Check initialization.
        upsampling_weight = _get_upsampling_weight(
            model.n_classes, model.n_classes, kernel_size=module.kernel_size[0]
        )
        is_success = torch.allclose(upsampling_weight, module.weight)
        if bilinear:
            assert is_success
        else:
            assert not is_success


def test_convolutionizedvgg16():
    vgg = ConvolutionizedVGG16()
    with torch.no_grad():
        random_tensor = torch.randn((1, 3, 500, 500))
        _ = vgg(random_tensor)
    torch_vgg = vgg16(weights=None)
    vgg.copy_weights_from_torchvision(torch_vgg)

    # Check features transplantation.
    conved_convs = _lazy_conv_params(vgg)
    torch_convs = _lazy_packed_conv_params(torch_vgg)
    for c1, c2 in zip(conved_convs, torch_convs):
        assert torch.allclose(c1, c2)

    # Check convolutionized FC layers.
    for source_index, conved_fc in zip([0, 3], [vgg.fc1, vgg.fc2]):
        source = torch_vgg.classifier[source_index]
        for src_param, conved_param in zip(source.parameters(), conved_fc.parameters()):
            assert torch.allclose(src_param.view(conved_param.shape), conved_param)


@pytest.mark.parametrize(
    "bilinear, final_trainable", product([True, False], [True, False])
)
def test_fcn32s(bilinear, final_trainable, get_test_config):
    config = deepcopy(get_test_config)
    config.model.trainable_final_upsampling = final_trainable
    config.model.bilinear_upsampling_init = bilinear
    fcn32s = FCN32s(config)
    assert fcn32s.n_classes == config.model.n_classes
    with torch.no_grad():
        random_tensor = torch.randn((1, 3, 224, 224))
        out = fcn32s(random_tensor)
    assert tuple(out.shape) == (1, config.model.n_classes, 224, 224)

    # Let's check initialization and trainability.
    _check_upsamplings(fcn32s, bilinear, final_trainable, False)


@pytest.mark.parametrize(
    "bilinear, final_trainable, inter_trainable",
    product([True, False], [True, False], [True, False]),
)
def test_fcn16s(bilinear, final_trainable, inter_trainable, get_test_config):
    config = deepcopy(get_test_config)
    config.model.trainable_final_upsampling = final_trainable
    config.model.bilinear_upsampling_init = bilinear
    config.model.trainable_intermediate_upsampling = inter_trainable
    fcn32s_state_dict = FCN32s(config).state_dict()
    fcn16s = FCN16s(config)
    fcn16s.load_weights_from_prev(fcn32s_state_dict)

    # Check initialization with the previous model.
    for param_name, param_tensor in fcn16s.named_parameters():
        if (fcn32_param := fcn32s_state_dict.get(param_name, None)) is not None and (
            fcn32_param.shape == param_tensor.shape
        ):
            assert torch.allclose(param_tensor, fcn32_param)

    assert fcn16s.n_classes == config.model.n_classes
    with torch.no_grad():
        random_tensor = torch.randn((1, 3, 224, 224))
        out = fcn16s(random_tensor)
    assert tuple(out.shape) == (1, config.model.n_classes, 224, 224)

    # Let's check initialization and trainability.
    _check_upsamplings(fcn16s, bilinear, final_trainable, inter_trainable)


@pytest.mark.parametrize(
    "bilinear, final_trainable, inter_trainable",
    product([True, False], [True, False], [True, False]),
)
def test_fcn8s(bilinear, final_trainable, inter_trainable, get_test_config):
    config = deepcopy(get_test_config)
    config.model.trainable_final_upsampling = final_trainable
    config.model.bilinear_upsampling_init = bilinear
    config.model.trainable_intermediate_upsampling = inter_trainable
    fcn16s_state_dict = FCN16s(config).state_dict()
    fcn8s = FCN8s(config)
    fcn8s.load_weights_from_prev(fcn16s_state_dict)

    # Check initialization with the previous model.
    for param_name, param_tensor in fcn8s.named_parameters():
        if (fcn16_param := fcn16s_state_dict.get(param_name, None)) is not None and (
            fcn16_param.shape == param_tensor.shape
        ):
            assert torch.allclose(param_tensor, fcn16_param)

    assert fcn8s.n_classes == config.model.n_classes
    with torch.no_grad():
        random_tensor = torch.randn((1, 3, 224, 224))
        out = fcn8s(random_tensor)
    assert tuple(out.shape) == (1, config.model.n_classes, 224, 224)

    # Let's check initialization and trainability.
    _check_upsamplings(fcn8s, bilinear, final_trainable, inter_trainable)
