import addict
import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

from source.modules import conv_layer, upsampling_layer

from .backbones import ConvolutionizedVGG16


class FCN8s(nn.Module):
    def __init__(self, config: addict.Dict) -> None:
        super().__init__()
        self.n_classes = config.model.n_classes
        self.vgg = ConvolutionizedVGG16()
        self.vgg.copy_weights_from_torchvision(vgg16(weights=VGG16_Weights.DEFAULT))

        # Stride 32 output of FCN:
        # Conv 1x1 to obtain class scores + Upsampling to source
        # image size.
        self.score_stride_32 = conv_layer(4096, self.n_classes, 1)
        self.stride_32_up = upsampling_layer(
            self.n_classes,
            self.n_classes,
            kernel_size=4,
            stride=2,
            bias=False,
            bilinear=config.model.bilinear_upsampling_init,
            trainable=config.model.trainable_intermediate_upsampling,
        )

        # Stride 16 output of FCN:
        # Conv 1x1 to obtain class scores from pool4 + Upsampling
        # to source image size.
        self.score_stride_16 = conv_layer(512, self.n_classes, 1)
        self.stride_16_up = upsampling_layer(
            self.n_classes,
            self.n_classes,
            kernel_size=4,
            stride=2,
            bias=False,
            bilinear=config.model.bilinear_upsampling_init,
            trainable=config.model.trainable_intermediate_upsampling,
        )

        # Stride 8 output of FCN:
        # Conv 1x1 to obtain class scores from pool3 + Upsampling
        # to source image size.
        self.score_stride_8 = conv_layer(256, self.n_classes, 1)
        self.final_up = upsampling_layer(
            self.n_classes,
            self.n_classes,
            kernel_size=16,
            stride=8,
            bias=False,
            bilinear=config.model.bilinear_upsampling_init,
            trainable=config.model.trainable_final_upsampling,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        *_, height, width = tensor.shape
        stride_32, stride_16, stride_8 = self.vgg(tensor)
        stride_32 = self.stride_32_up(self.score_stride_32(stride_32))
        stride_16 = self.score_stride_16(stride_16)[
            :, :, 5 : 5 + stride_32.shape[2], 5 : 5 + stride_32.shape[3]
        ]
        stride_32 += stride_16
        stride_32 = self.stride_16_up(stride_32)
        stride_8 = self.score_stride_8(stride_8)[
            :, :, 9 : 9 + stride_32.shape[2], 9 : 9 + stride_32.shape[3]
        ]
        stride_32 += stride_8
        return self.final_up(stride_32)[
            :, :, 31 : 31 + height, 31 : 31 + width
        ].contiguous()

    def load_weights_from_prev(self, prev_ckpt: dict[str, torch.Tensor]) -> None:
        if prev_ckpt is None:
            return
        for param_name, param_tensor in self.named_parameters():
            if (fcn32_param := prev_ckpt.get(param_name)) is not None and (
                fcn32_param.shape == param_tensor.shape
            ):
                param_tensor.data.copy_(fcn32_param.data)
