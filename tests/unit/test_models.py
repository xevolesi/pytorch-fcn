from copy import deepcopy
from itertools import product

import pytest
import torch
from torchvision.models import vgg16

from source.models import FCN, TimmFCN
from source.models.backbones import ConvolutionizedVGG16, TimmBackbone
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


def test_timm_backbone(get_test_config):
    backbone = TimmBackbone(
        get_test_config.model.backbone_name,
        get_test_config.model.in_chans,
        pretrained=False,
    )
    tensor = torch.randn((1, 3, 500, 500))
    with torch.no_grad():
        outputs = backbone(tensor)
        assert isinstance(outputs, list)
        assert len(outputs) >= 3


@pytest.mark.parametrize(
    "stride, bilinear, final_trainable, inter_trainable",
    product([32, 16, 8], [True, False], [True, False], [True, False]),
)
def test_fcn(stride, bilinear, final_trainable, inter_trainable, get_test_config):
    paper_model_cfg = dict(
        stride=stride,
        trainable_final_upsampling=final_trainable,
        bilinear_upsampling_init=bilinear,
        trainable_intermediate_upsampling=inter_trainable,
        n_classes=get_test_config.model.n_classes,
    )
    model = FCN(**paper_model_cfg)

    match paper_model_cfg["stride"]:
        case 32:
            prev_strided_model = None
        case 16:
            cfg = deepcopy(paper_model_cfg)
            cfg["stride"] = 32
            prev_strided_model = FCN(**cfg).state_dict()
        case 8:
            cfg = deepcopy(paper_model_cfg)
            cfg["stride"] = 16
            prev_strided_model = FCN(**cfg).state_dict()
    model.load_weights_from_prev(prev_strided_model)

    # Check initialization with the previous model.
    if prev_strided_model is None:
        assert model.stride == 32
    else:
        for param_name, param_tensor in model.named_parameters():
            if (
                prev_strided_param := prev_strided_model.get(param_name)
            ) is not None and (prev_strided_param.shape == param_tensor.shape):
                assert torch.allclose(param_tensor, prev_strided_param)

    assert model.n_classes == get_test_config.model.n_classes
    with torch.no_grad():
        random_tensor = torch.randn((1, 3, 224, 224))
        out = model(random_tensor)
    assert tuple(out.shape) == (1, get_test_config.model.n_classes, 224, 224)

    # Let's check initialization and trainability.
    _check_upsamplings(model, bilinear, final_trainable, inter_trainable)


@pytest.mark.parametrize(
    "stride, backbone_name, in_chans",
    product(
        [32, 16, 8],
        ["densenet121", "ecaresnext50t_32x4d", "tf_efficientnetv2_s"],
        [3, 6, 9],
    ),
)
def test_timm_fcn(stride, backbone_name, in_chans, get_test_config):
    timm_cfg = dict(
        stride=stride,
        backbone_name=backbone_name,
        pretrained=False,
        in_chans=in_chans,
        n_classes=get_test_config.model.n_classes,
    )
    model = TimmFCN(**timm_cfg)

    match timm_cfg["stride"]:
        case 32:
            prev_strided_model = None
        case 16:
            cfg = deepcopy(timm_cfg)
            cfg["stride"] = 32
            prev_strided_model = TimmFCN(**cfg).state_dict()
        case 8:
            cfg = deepcopy(timm_cfg)
            cfg["stride"] = 16
            prev_strided_model = TimmFCN(**cfg).state_dict()
    model.load_weights_from_prev(prev_strided_model)

    # Check initialization with the previous model.
    if prev_strided_model is None:
        assert model.stride == 32
    else:
        for param_name, param_tensor in model.named_parameters():
            if (
                prev_strided_param := prev_strided_model.get(param_name)
            ) is not None and (prev_strided_param.shape == param_tensor.shape):
                assert torch.allclose(param_tensor, prev_strided_param)

    assert model.n_classes == get_test_config.model.n_classes
    with torch.no_grad():
        random_tensor = torch.randn((1, timm_cfg["in_chans"], 224, 224))
        out = model(random_tensor)
    assert tuple(out.shape) == (1, get_test_config.model.n_classes, 224, 224)
