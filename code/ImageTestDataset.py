import os
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageOps
import torch


class ImageTestDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir  # 添加标签路径
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 获取图像路径
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        # 加载标签路径并读取第一行第一个数字作为标签
        label_path = os.path.join(self.label_dir, image_name.replace('.jpeg', '.txt'))
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            label = int(first_line.split()[0])  # 提取第一个数字

        # 检查标签范围，假设标签为 1-6
        # assert 0 <= label <= 1, f"Invalid label value: {label}"

        # 转换为张量,使标签从 0 开始
        label = torch.tensor(label, dtype=torch.long)

        # 使用重采样滤波器调整图像大小
        image = ImageOps.fit(image, (224, 224), method=Image.LANCZOS)

        # 如果有图像变换，应用变换
        if self.transform:
            image = self.transform(image)

        return image, label