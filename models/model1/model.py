import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class LocModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        # фризим слои, обучать их не будем (хотя технически можно)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.clf = nn.Sequential(
            nn.Linear(512*8*8, 128),
            nn.Sigmoid(),
            nn.Linear(128, 3)
        )

        self.box = nn.Sequential(
            nn.Linear(512*8*8, 128),
            nn.Sigmoid(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, img):
        resnet_out = self.feature_extractor(img)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)
        pred_classes = self.clf(resnet_out)
        pred_boxes = self.box(resnet_out)
        return pred_classes, pred_boxes