import torch
from torch import nn
from torchvision.models import resnet18


class ResNet_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(pretrained=False)
        # заменяем слой
        self.model.fc = nn.Linear(512, 4)
    def forward(self, x):
        return self.model(x)