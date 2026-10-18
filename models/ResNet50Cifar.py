import torch.nn as nn
from torchvision.models import resnet50


class ResNet50Cifar(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"ResNet50Cifar expects 3 input channels, got {in_channels}")
        model = resnet50(weights=None)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)


def resnet50_cifar(num_classes=10, in_channels=3):
    return ResNet50Cifar(num_classes=num_classes, in_channels=in_channels)
