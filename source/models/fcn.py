import addict
from torch import nn

from source.modules import conv_layer, upsampling_layer


class FCN32VGG16(nn.Module):
    def __init__(self, config: addict.Dict) -> None:
        super().__init__()
        self.n_classes = config.model.n_classes

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
        self.upscore = upsampling_layer(
            self.n_classes,
            self.n_classes,
            kernel_size=64,
            stride=32,
            bias=False,
            bilinear=config.model.bilinear_upsampling_init,
            trainable=config.model.trainable_upsampling,
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
