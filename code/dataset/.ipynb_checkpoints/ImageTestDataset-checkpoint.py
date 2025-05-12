import os
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageOps

class ImageTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = Image.open(os.path.join(self.image_dir, image_name))

        image = Image.open(image_name).convert('RGB')
        # 使用重采样滤波器
        image= ImageOps.fit(image,(224,224), method= Image.LANCZOS)

        if self.transform:
            image = self.transform(image)

        return image