import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from dataset.ImageTestDataset import ImageTestDataset
from model.SegEfficientNet import EfficientNetModel
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_model(model, test_loader, device):
    model.eval()
    total_num = 0
    all_boundary_predictions = []
    all_direction_predictions = []
    all_shape_predictions = []
    print(">>>>>>>>>>>>>>>feature test time <<<<<<<<<<<<<<<<<<<<<<")

    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            boundary_out, direction_out, shape_out = model(images)
            boundary_pred = torch.max(boundary_out, 1)
            direction_pred = torch.max(direction_out, 1)
            shape_pred = torch.max(shape_out, 1)

            total_num += images.size(0)
            all_boundary_predictions.extend(boundary_pred.cpu().numpy())
            all_direction_predictions.extend(direction_pred.cpu().numpy())
            all_shape_predictions.extend(shape_pred.cpu().numpy())

        # id列
        id_column = list(range(1, total_num + 1))

        # 将每列的结果保存
        df = pd.DataFrame({
            'id': id_column,
            'boundary': all_boundary_predictions,
            'direction': all_direction_predictions,
            'shape': all_shape_predictions
        })
        df.to_csv('./feature_pred.csv', index=False)
        return

def run_detect_feature(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试数据集
    test_dataset = ImageTestDataset(image_dir=args.fea_test_dir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = EfficientNetModel().to(device)

    if os.path.exists(args.fea_model_path):
        model.load_state_dict(torch.load(args.fea_model_path))
        model.eval()
        evaluate_model(model, test_dataloader, device)
