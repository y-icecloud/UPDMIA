import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageTrainDataSet(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __getitem__(self, index):

        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace('_cropped.jpg', '.txt'))


        images = Image.open(image_path).convert('RGB')

        with open(label_path, 'r') as f:
            labels = list(map(int,f.read().strip().split(' ')[:4]))
        labels = torch.tensor(labels)

        if self.transform:
            images = self.transform(images)


        return images, labels

    def __len__(self):
        return len(self.image_files)