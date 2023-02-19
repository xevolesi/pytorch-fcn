from copy import deepcopy

import addict
import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

from source.models.utils import generate_bilinear_kernel, linear2conv2d
from source.modules import SpatialCaffeLikeCrop


class FCN32VGG16(nn.Module):
    def __init__(self, config: addict.Dict) -> None:
        super().__init__()
        self.n_classes = config.model.n_classes
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)

        # As authors suggest in paper we need to keep VGG16 feature
        # extractor as is.
        self.features = deepcopy(vgg.features)

        # Except that we need to change the first conv padding:
        # Padding is 100 in first convolution to be able to upsample
        # penultimate score feature map to target image size. This
        # logic was directly copy from authors source code here:
        # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/train.prototxt#L27
        self.features[0].padding = (100, 100)

        # Now we need to convolutionized FC layers in classifier
        # as suggested in paper. In VGG16 classifier there are 2
        # droput layers between FC layers and authors also kept it.
        # We change it to Dropout2d layer since we are trying to
        # convolutionized FC layers.
        self.convolutionized = nn.Sequential(
            linear2conv2d(vgg.classifier[0], (7, 7)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5, inplace=True),
            linear2conv2d(vgg.classifier[3], (1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5, inplace=True),
        )

        # Okay, now we need to add 1x1 convolution to compute scores
        # for n_classes as suggested in paper.
        self.scorer = nn.Conv2d(
            in_channels=4096, out_channels=self.n_classes, kernel_size=(1, 1)
        )

        # Authors also transplant old score layers from last
        # FC classification layer of VGG16. They took first
        # `n_classes` rows of FC layer weights and transplant it
        # in convolution kernel: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/surgery.py#L61
        if config.model.transplant_score_layer:
            self._transplant_old_score_layer(last_vgg16_layer=vgg.classifier[-1])

        # And now we need to upsample scores' map to the target
        # image size: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/train.prototxt#L494
        self.up = nn.ConvTranspose2d(
            in_channels=self.n_classes,
            out_channels=self.n_classes,
            kernel_size=64,
            stride=32,
            bias=False,
        )

        # And now we need to crop out meaningfull part of our feature
        # map along height and width.
        self.crop = SpatialCaffeLikeCrop(offset=(19, 19))

        # In paper all upsampling layers were initialized with bilinear
        # upsampling kernel. And only last is trainable.
        if config.model.init_upsampling_as_bilinear:
            self._initialize_upsampling_layers()

        # We don't need it anymore.
        del vgg

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.features(tensor)
        logits = self.convolutionized(features)
        scores = self.scorer(logits)
        up = self.up(scores)
        return self.crop(to_crop=up, reference=tensor)

    def _initialize_upsampling_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.copy_(
                    generate_bilinear_kernel(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size[0],
                        dtype=module.weight.dtype,
                    )
                )

    def _transplant_old_score_layer(self, last_vgg16_layer: nn.Linear) -> None:
        scorer_shape = self.scorer.weight.shape[0]
        self.scorer.weight.data.copy_(
            last_vgg16_layer.weight.data[:scorer_shape, ...].view(
                self.scorer.weight.shape
            )
        )
