from torch import nn
from torchvision.models import resnet18


def build_model():
    """ResNet18 trained from scratch, adapted to CIFAR's 32x32 images."""
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
