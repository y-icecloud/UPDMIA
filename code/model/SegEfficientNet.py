from torch import nn
from torchvision import models
from model.classifier import FeatureClassifier
import os

class EfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
        # 使用预训练权重
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier = nn.Identity()
        self.classifier = FeatureClassifier()
        
    def forward(self, x):
        # 首先提取特征
        x = self.efficientnet.features(x)
        # 然后再特征提取
        boundary_out, direction_out, shape_out = self.classifier(x)
        return boundary_out, direction_out, shape_out


