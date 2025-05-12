import os
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageOps

class DetectDataSet(Dataset):
    def __init__(self, image_dir, transform=None, img_name = ''):
        self.image_dir = image_dir
        self.transform = transform
        allImages = os.listdir(image_dir)
        if (len(img_name) == 0):
         self.images = allImages
        else:
         print(img_name)
         self.images = [img for img in allImages if img == img_name]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        image = ImageOps.fit(image, (224, 224), method=Image.LANCZOS)

        if self.transform:
            image = self.transform(image)

        return image