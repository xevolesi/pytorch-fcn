import addict
from torch import nn


class FCN32VGG16(nn.Module):
    def __init__(self, config: addict.Dict) -> None:
        super().__init__()
        self.n_classes = config.model.n_classes

        # Padding is 100 in first convolution to be able to upsample
        # penultimate score feature map to target image size. This
        # logic was directly copy from authors source code here:
        # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/train.prototxt#L27
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (100, 100))
        self.act1 = nn.ReLU(inplace=True)
