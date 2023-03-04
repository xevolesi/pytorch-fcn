import torch
from torch import nn


class ConvolutionizedVGG16(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Stride 2.
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # Stride 4.
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # Stride 8.
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # Stride 16.
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # Stride 32.
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # Convolutionized classifier's FC layers.
        self.fc1 = nn.Conv2d(512, 4096, 7, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d()
        self.fc2 = nn.Conv2d(4096, 4096, 1, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.relu1_1(self.conv1_1(tensor))
        tensor = self.relu1_2(self.conv1_2(tensor))
        tensor = self.pool1(tensor)

        tensor = self.relu2_1(self.conv2_1(tensor))
        tensor = self.relu2_2(self.conv2_2(tensor))
        tensor = self.pool2(tensor)

        tensor = self.relu3_1(self.conv3_1(tensor))
        tensor = self.relu3_2(self.conv3_2(tensor))
        tensor = self.relu3_3(self.conv3_3(tensor))
        stride_8_features = self.pool3(tensor)

        tensor = self.relu4_1(self.conv4_1(stride_8_features))
        tensor = self.relu4_2(self.conv4_2(tensor))
        tensor = self.relu4_3(self.conv4_3(tensor))
        stride_16_features = self.pool4(tensor)

        tensor = self.relu5_1(self.conv5_1(stride_16_features))
        tensor = self.relu5_2(self.conv5_2(tensor))
        tensor = self.relu5_3(self.conv5_3(tensor))
        tensor = self.pool5(tensor)

        tensor = self.relu1(self.fc1(tensor))
        tensor = self.drop1(tensor)

        tensor = self.relu2(self.fc2(tensor))
        tensor = self.drop2(tensor)

        return tensor, stride_16_features, stride_8_features

    def copy_weights_from_torchvision(self, vgg: nn.Module) -> None:
        # Copy weights from feature extractor.
        features = [
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.conv4_1,
            self.conv4_2,
            self.conv4_3,
            self.conv5_1,
            self.conv5_2,
            self.conv5_3,
        ]
        for source, target in zip(
            filter(lambda module: isinstance(module, nn.Conv2d), vgg.features), features
        ):
            target.weight.data.copy_(source.weight.data)
            target.bias.data.copy_(source.bias.data)

        # Copy weights from classifier FC layers to convolutionized
        # FC layers.
        for source_index, target in zip([0, 3], [self.fc1, self.fc2]):
            source = vgg.classifier[source_index]
            target.weight.data.copy_(source.weight.data.view(target.weight.shape))
            target.bias.data.copy_(source.bias.data.view(target.bias.shape))
