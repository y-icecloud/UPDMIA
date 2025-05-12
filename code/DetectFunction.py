import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from ImageTestDataset import ImageTestDataset  # 使用带标签的数据集
from EfficientNetModel import EfficientNetModel
import os
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# 标准化参数与ImageNet一致
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_model(model, test_loader, device, args):
    model.eval()
    total_num = 0
    res = []
    all_labels = []
    all_preds = []
    
    print(">>>>>>>>>>>>>>> Start Evaluation <<<<<<<<<<<<<<<<<<<<<<")
    
    with torch.no_grad():
        for images, labels in test_loader:  # 同时获取图像和标签
            images = images.to(device)
            labels = labels.to(device)
            # 保存真实标签
            all_labels.extend(labels.cpu().numpy())
            
            # 前向传播
            pred = model(images)
            if pred is None:
                print("警告：模型输出为 None")
                print(f"输入形状: {images.shape}")
                print(f"输入类型: {type(images)}")
            pred_class = torch.argmax(pred, dim=1)
            
            # 保存预测结果
            all_preds.extend(pred_class.cpu().numpy())
            
            # 生成诊断描述
            probability = F.softmax(pred, dim=1)
            probability = probability.detach().cpu().numpy()
            total_num += images.size(0)
            
#             # 根据类别生成诊断结果
#             if args.category == "MRI":
#                 if pred_class.item() == 1:
#                     temp = f"患癌概率：{(probability[0][1]*100):.2f}%"
#                 else:
#                     temp = "健康，需定期检查"
#             elif args.category == "CT":
#                 class_names = ["健康", "鳞状细胞癌", "大细胞癌", "腺癌"]
#                 temp = f"预测类别：{class_names[pred_class.item()]} ({probability[0][pred_class.item()]*100:.2f}%)"
#             # 其他类别类似处理...
            
#             res.append(temp)
#             print(f"诊断结果: {temp}")

    # 计算指标
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n=============== 评估指标 ===============")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 保存结果到CSV
    df = pd.DataFrame({
        'id': list(range(1, total_num+1)),
        'diagnosis': res,
        'true_label': all_labels,
        'pred_label': all_preds
    })
    df.to_csv('./pred_with_metrics.csv', index=False)
    print("结果已保存至 pred_with_metrics.csv")

def run_detect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用带标签的测试集
    test_image_path = os.path.join(args.cla_test_dir, 'images')
    test_label_path = os.path.join(args.cla_test_dir, 'labels')  # 确保标签路径正确
    test_dataset = ImageTestDataset(test_image_path, test_label_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 初始化模型
    model = EfficientNetModel(args).to(device)
    model.load_state_dict(torch.load(f"{args.category}_model.pth", map_location=device))
    model.eval()
    
    evaluate_model(model, test_dataloader, device, args)