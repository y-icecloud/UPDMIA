from tqdm import tqdm
import torch
from model.SegEfficientNet import EfficientNetModel
from torch import nn
from dataset.ImageTrainDataset import ImageTrainDataSet
from torchvision import transforms
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([
    transforms.RandomCrop(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_train_feature(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetModel().to(device)
    
    if os.path.exists(args.fea_model_path):
        model.load_state_dict(torch.load(args.fea_model_path))
        
    model.train()
    running_loss = 0.0
    b_correct = 0
    d_correct = 0
    s_correct = 0
    total = 0
    loss = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)

    train_image_path = args.fea_train_dir + '/images'
    train_label_path = args.fea_train_dir + '/labels'
    train_dataset = ImageTrainDataSet(train_image_path, train_label_path, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    print(">>>>>>>>>>>>>>>>>>>>>> feature training time <<<<<<<<<<<<<<<<<<<<<<<<")
    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        
        labels[:,0][labels[:,0] == 999] = 0
        labels[:,1][labels[:,1] == 999] = 0
        labels[:,2][labels[:,2] == 999] = 0

        optimizer.zero_grad()
        boundary_out, direction_out, shape_out= model(images)


        boundary_loss = criterion(boundary_out, labels[:,0])
        direction_loss = criterion(direction_out, labels[:, 1])
        shape_loss = criterion(shape_out, labels[:, 2])
        loss = boundary_loss + direction_loss + shape_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, b_predicted = torch.max(boundary_out, 1)
        b_correct += (b_predicted == labels[:, 0]).sum().item()

        _, d_predicted = torch.max(direction_out, 1)
        d_correct += (d_predicted == labels[:, 1]).sum().item()

        _, s_predicted = torch.max(shape_out, 1)
        s_correct += (s_predicted == labels[:, 2]).sum().item()
        
        total += labels.size(0)

    epoch_loss = running_loss / len(train_dataloader)

    b_accuracy = 100 * b_correct / total
    d_accuracy = 100 * d_correct / total
    s_accuracy = 100 * s_correct / total
    print(f'Train Loss: {epoch_loss:.4f}, Boundary Train Accuracy: {b_accuracy:.2f}%')
    print(f'Train Loss: {epoch_loss:.4f}, Direction Train Accuracy: {d_accuracy:.2f}%')
    print(f'Train Loss: {epoch_loss:.4f}, Shape Train Accuracy: {s_accuracy:.2f}%')