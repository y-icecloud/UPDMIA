import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.batch_norm = nn.BatchNorm2d(1280)
        self.swish = nn.SiLU(inplace=False)
        self.dropout = nn.Dropout(p=0.5,inplace=False)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x  = self.batch_norm(x)
        x = self.swish(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class FeatureClassifier(nn.Module):
    def __init__(self):
        super(FeatureClassifier, self).__init__()
        self.batch_norm = nn.BatchNorm2d(1280)
        self.swish = nn.SiLU(inplace=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5, inplace=False)

        # 边缘
        self.boundary_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, 2)
        )

        # 方位
        self.direction_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, 2)
        )

        # 形状
        self.shape_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4,inplace=False),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        # 对每个特征进行独立的二分类
        boundary_out = self.boundary_head(x)
        direction_out = self.direction_head(x)
        shape_out = self.shape_head(x)

        return boundary_out, direction_out, shape_out