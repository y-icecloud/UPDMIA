import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from ImageTestDataset import ImageTestDataset
from ImageTrainDataset import ImageTrainDataset
from EfficientNetModel import EfficientNetModel

def run_train_classification(args):
    # define transform
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetModel(args).to(device)

    # 加入学习率调度
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_image_path = os.path.join(args.cla_train_dir, 'images')
    train_label_path = os.path.join(args.cla_train_dir, 'labels')
    train_dataset = ImageTrainDataset(train_image_path, train_label_path, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # 加了个验证集代码方便后续使用
    val_image_path = os.path.join(args.cla_val_dir, 'images') if hasattr(args, 'cla_val_dir') else None
    val_label_path = os.path.join(args.cla_val_dir, 'labels') if hasattr(args, 'cla_val_dir') else None
    val_dataloader = None
    if val_image_path and val_label_path:
        val_dataset = ImageTrainDataset(val_image_path, val_label_path, transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print("-------------------- Start Classification Training ---------------------")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_total = 0
        train_correct = 0
        
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()

            # 加一个梯度裁剪防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, train_pred = torch.max(predictions, 1)
            train_correct += (train_pred == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_accuracy = 100 * train_correct / train_total
        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy:{train_accuracy:.2f}%")
        
        # 下面的如果有验证集可以跑
        if val_dataloader:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_dataloader:
                    images, labels = images.to(device), labels.to(device)

                    predictions = model(images)
                    loss = criterion(predictions, labels)
                    val_loss += loss.item() * images.size(0)

                    _, predicted = torch.max(predictions, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            val_loss /= len(val_dataloader.dataset)
            accuracy = 100 * correct / total

            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Save the model periodically
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"{args.category}_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")
    print("-------------------- Training Complete ---------------------")


def run_test_classification(args):
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetModel(args).to(device)
    # model.load_state_dict(torch.load(f"./working/{args.category}_model_epoch_{args.epochs}.pth"))
    model.load_state_dict(torch.load(f"./working/{args.category}_model_epoch_{args.epochs}.pth", map_location=torch.device('cpu')))
    model.eval()
    
    test_image_path = os.path.join(args.cla_test_dir, 'images')
    test_label_path = os.path.join(args.cla_test_dir, 'labels')
    test_dataset = ImageTestDataset(test_image_path, test_label_path,transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        incorrect = []
        for image, label in tqdm(test_dataloader):
            image, label = image.to(device), label.to(device)
            predictions = model(image)
            loss = criterion(predictions, label)
            
            running_loss += loss.item() * image.size(0)
            _, predicted = torch.max(predictions, 1)
            correct += (label == predicted).sum().item()
            incorrect.append(label != predicted)
            total += label.size(0)
        
        running_loss /= len(test_dataloader.dataset)
        accuracy = 100 * correct/total
        print(f"Test Loss:{running_loss:.4f}, Accuracy:{accuracy:.2f}%")