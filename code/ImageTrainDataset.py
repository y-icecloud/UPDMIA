import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageTrainDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __getitem__(self, index):
        # 获取路径
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace('.jpeg', '.txt'))

        # 加载图像
        images = Image.open(image_path).convert('RGB')

        # 读取标签文件并提取第一行的第一个数字
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()  # 读取第一行
            label = int(first_line.split()[0])  # 提取第一行的第一个数字

        # 检查标签范围是否在 1 到 6 之间
        # assert 0 <= label <= 1, f"Invalid label value: {label}"

        # 转换为 PyTorch 张量并减去 1（因为 PyTorch 中的 CrossEntropyLoss 标签从 0 开始）
        labels = torch.tensor(label, dtype=torch.long)

        if self.transform:
            images = self.transform(images)

        return images, labels

    def __len__(self):
        return len(self.image_files)