import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

from source.modules import FCNHead, conv_layer, upsampling_layer

from .backbones import ConvolutionizedVGG16, TimmBackbone


class FCN(nn.Module):
    _offsets: dict[int, tuple[int, int, int]] = {
        32: (0, 0, 19),
        16: (0, 5, 27),
        8: (9, 5, 31),
    }

    def __init__(
        self,
        stride: int,
        n_classes: int,
        trainable_intermediate_upsampling: bool = False,
        trainable_final_upsampling: bool = False,
        bilinear_upsampling_init: bool = True,
    ) -> None:
        super().__init__()
        if stride not in self._offsets:
            raise ValueError(
                f"Expected `stride` to be one of (8, 16, 32), but got {stride}"
            )
        self.n_classes = n_classes
        self.stride = stride
        self.vgg = ConvolutionizedVGG16()
        self.vgg.copy_weights_from_torchvision(vgg16(weights=VGG16_Weights.DEFAULT))

        self.score_stride_32 = conv_layer(4096, self.n_classes, 1)
        self.stride_32_up = nn.Identity()
        self.score_stride_16 = nn.Identity()
        if self.stride < 32:
            self.stride_32_up = upsampling_layer(
                self.n_classes,
                self.n_classes,
                kernel_size=4,
                stride=2,
                bias=False,
                bilinear=bilinear_upsampling_init,
                trainable=trainable_intermediate_upsampling,
            )
            self.score_stride_16 = conv_layer(512, self.n_classes, 1)

        self.stride_16_up = nn.Identity()
        self.score_stride_8 = nn.Identity()
        if self.stride < 16:
            self.stride_16_up = upsampling_layer(
                self.n_classes,
                self.n_classes,
                kernel_size=4,
                stride=2,
                bias=False,
                bilinear=bilinear_upsampling_init,
                trainable=trainable_intermediate_upsampling,
            )
            self.score_stride_8 = conv_layer(256, self.n_classes, 1)

        self.final_up = upsampling_layer(
            self.n_classes,
            self.n_classes,
            kernel_size=self.stride * 2,
            stride=self.stride,
            bias=False,
            bilinear=bilinear_upsampling_init,
            trainable=trainable_final_upsampling,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        *_, height, width = tensor.shape
        stride_32, stride_16, stride_8 = self.vgg(tensor)
        stride_32 = self.stride_32_up(self.score_stride_32(stride_32))
        if self.stride < 32:
            stride_16 = self.score_stride_16(stride_16)[
                :,
                :,
                self._offsets[self.stride][1] : self._offsets[self.stride][1]
                + stride_32.shape[2],
                self._offsets[self.stride][1] : self._offsets[self.stride][1]
                + stride_32.shape[3],
            ]
            stride_32 += stride_16
            stride_32 = self.stride_16_up(stride_32)
        if self.stride < 16:
            stride_8 = self.score_stride_8(stride_8)[
                :,
                :,
                self._offsets[self.stride][0] : self._offsets[self.stride][0]
                + stride_32.shape[2],
                self._offsets[self.stride][0] : self._offsets[self.stride][0]
                + stride_32.shape[3],
            ]
            stride_32 += stride_8
        return self.final_up(stride_32)[
            :,
            :,
            self._offsets[self.stride][2] : self._offsets[self.stride][2] + height,
            self._offsets[self.stride][2] : self._offsets[self.stride][2] + width,
        ].contiguous()

    def load_weights_from_prev(self, prev_ckpt: dict[str, torch.Tensor]) -> None:
        if prev_ckpt is None:
            return
        for param_name, param_tensor in self.named_parameters():
            if (prev_param := prev_ckpt.get(param_name)) is not None and (
                prev_param.shape == param_tensor.shape
            ):
                param_tensor.data.copy_(prev_param.data)


class TimmFCN(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        in_chans: int,
        stride: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = TimmBackbone(backbone_name, in_chans, pretrained)
        self.n_classes = n_classes
        self.stride = stride

        self.score_stride_8 = nn.Identity()
        self.score_stride_16 = nn.Identity()
        self.score_stride_32 = FCNHead(self.backbone.out_channels[-1], self.n_classes)
        if self.stride < 32:
            self.score_stride_16 = FCNHead(
                self.backbone.out_channels[-2], self.n_classes
            )
        if self.stride < 16:
            self.score_stride_8 = FCNHead(self.backbone.out_channels[-3], self.n_classes)

    def forward(self, tensor):
        *_, height, width = tensor.shape
        features = self.backbone(tensor)
        stride_32 = self.score_stride_32(features[-1])
        if self.stride < 32:
            stride_16 = self.score_stride_16(features[-2])
            stride_32 = nn.functional.interpolate(
                stride_32, stride_16.shape[-2:], mode="bilinear", align_corners=False
            )
            stride_32 += stride_16
        if self.stride < 16:
            stride_8 = self.score_stride_8(features[-3])
            stride_32 = nn.functional.interpolate(
                stride_32, stride_8.shape[-2:], mode="bilinear", align_corners=False
            )
            stride_32 += stride_8
        return nn.functional.interpolate(
            stride_32, size=(height, width), mode="bilinear", align_corners=False
        )

    def load_weights_from_prev(self, prev_ckpt: dict[str, torch.Tensor]) -> None:
        if prev_ckpt is None:
            return
        for param_name, param_tensor in self.named_parameters():
            if (prev_param := prev_ckpt.get(param_name)) is not None and (
                prev_param.shape == param_tensor.shape
            ):
                param_tensor.data.copy_(prev_param.data)
