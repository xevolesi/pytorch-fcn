import addict
import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

from source.modules import conv_layer, upsampling_layer

from .backbones import ConvolutionizedVGG16


class FCN32s(nn.Module):
    def __init__(self, config: addict.Dict) -> None:
        super().__init__()
        self.n_classes = config.model.n_classes
        self.vgg = ConvolutionizedVGG16()
        self.vgg.copy_weights_from_torchvision(vgg16(weights=VGG16_Weights.DEFAULT))

        # Stride 32 output of FCN:
        # Conv 1x1 to obtain class scores + Upsampling to source image size.
        self.score_stride_32 = conv_layer(4096, self.n_classes, 1)
        self.final_up = upsampling_layer(
            self.n_classes,
            self.n_classes,
            kernel_size=64,
            stride=32,
            bias=False,
            bilinear=config.model.bilinear_upsampling_init,
            trainable=config.model.trainable_final_upsampling,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        *_, height, width = tensor.shape
        stride_32, *_ = self.vgg(tensor)
        stride_32 = self.final_up(self.score_stride_32(stride_32))
        return stride_32[:, :, 19 : 19 + height, 19 : 19 + width].contiguous()

    def load_weights_from_prev(self, prev_ckpt: dict[str, torch.Tensor]) -> None:
        return
