from torchvision import models
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.batch_norm = nn.BatchNorm2d(2048)
        self.swish = nn.SiLU(inplace=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5, inplace=False)

        # CT
        self.boundary_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, 4)
        )

        # MRI
        self.direction_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, 2)
        )

        # PET
        self.pet_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, 3)
        )
        
        # Slice
        self.slice_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, 3)
        )
        
        # X-Ray
        self.x_ray_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        if self.args.category == 'CT':
            CT_out = self.boundary_head(x)
            return CT_out
        elif self.args.category == 'MRI':
            MRI_out = self.direction_head(x)
            return MRI_out
        elif self.args.category == 'PET':
            PET_out = self.pet_head(x)
            return PET_out
        elif self.args.category == 'Slide':
            Slice_out = self.slice_head(x)
            return Slice_out
        elif self.args.category == 'X_Ray':
            X_Ray_out = self.x_ray_head(x)
            return X_Ray_out
        


class EfficientNetModel(nn.Module):
    def __init__(self,args):
        super(EfficientNetModel, self).__init__()
        # 使用预训练权重
        self.efficientnet = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier = nn.Identity()
        self.classifier = Classifier(args)

    def forward(self, x):
        x = self.efficientnet.features(x)
        res = self.classifier(x)
        return res  # 确保返回值不是 None