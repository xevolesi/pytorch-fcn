import numpy as np
import torch
from torch import nn


def conv_layer(in_channels, out_channles, kernel_size, stride=1, padding=0, bias=True):
    layer = nn.Conv2d(in_channels, out_channles, kernel_size, stride, padding, bias=bias)
    layer.weight.data.zero_()
    if bias:
        layer.bias.data.zero_()
    return layer


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    Make a 2D bilinear kernel suitable for unsampling
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (
        1 - abs(og[1] - center) / factor
    )
    weight = np.zeros(
        (in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32
    )
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight).float()


def bilinear_upsampling(in_channels, out_channels, kernel_size, stride, bias=False):
    initial_weight = get_upsampling_weight(in_channels, out_channels, kernel_size)
    layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
    layer.weight.data.copy_(initial_weight)
    layer.weight.requires_grad = False
    return layer


class FCN32VGG16(nn.Module):
    def __init__(self, n_classes: int = 21) -> None:
        super().__init__()
        self.n_classes = n_classes

        # VGG16.
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        self.fc1 = conv_layer(512, 4096, 7)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d()

        self.fc2 = conv_layer(4096, 4096, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d()

        self.score_fr = conv_layer(4096, self.n_classes, 1)
        self.upscore = bilinear_upsampling(
            self.n_classes, self.n_classes, 64, stride=32, bias=False
        )

    def forward(self, tensor):
        *_, height, width = tensor.shape

        # VGG16.
        tensor = self.relu1_1(self.conv1_1(tensor))
        tensor = self.relu1_2(self.conv1_2(tensor))
        tensor = self.pool1(tensor)

        tensor = self.relu2_1(self.conv2_1(tensor))
        tensor = self.relu2_2(self.conv2_2(tensor))
        tensor = self.pool2(tensor)

        tensor = self.relu3_1(self.conv3_1(tensor))
        tensor = self.relu3_2(self.conv3_2(tensor))
        tensor = self.relu3_3(self.conv3_3(tensor))
        tensor = self.pool3(tensor)

        tensor = self.relu4_1(self.conv4_1(tensor))
        tensor = self.relu4_2(self.conv4_2(tensor))
        tensor = self.relu4_3(self.conv4_3(tensor))
        tensor = self.pool4(tensor)

        tensor = self.relu5_1(self.conv5_1(tensor))
        tensor = self.relu5_2(self.conv5_2(tensor))
        tensor = self.relu5_3(self.conv5_3(tensor))
        tensor = self.pool5(tensor)

        tensor = self.relu1(self.fc1(tensor))
        tensor = self.drop1(tensor)

        tensor = self.relu2(self.fc2(tensor))
        tensor = self.drop2(tensor)

        tensor = self.score_fr(tensor)
        tensor = self.upscore(tensor)
        return tensor[:, :, 19 : 19 + height, 19 : 19 + width].contiguous()

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1,
            self.relu1_1,
            self.conv1_2,
            self.relu1_2,
            self.pool1,
            self.conv2_1,
            self.relu2_1,
            self.conv2_2,
            self.relu2_2,
            self.pool2,
            self.conv3_1,
            self.relu3_1,
            self.conv3_2,
            self.relu3_2,
            self.conv3_3,
            self.relu3_3,
            self.pool3,
            self.conv4_1,
            self.relu4_1,
            self.conv4_2,
            self.relu4_2,
            self.conv4_3,
            self.relu4_3,
            self.pool4,
            self.conv5_1,
            self.relu5_1,
            self.conv5_2,
            self.relu5_2,
            self.conv5_3,
            self.relu5_3,
            self.pool5,
        ]

        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ["fc1", "fc2"]):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
