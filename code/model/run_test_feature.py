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

def run_test_feature(args, best_loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetModel().to(device)

    model.eval()
    running_loss = 0.0
    b_correct = 0
    d_correct = 0
    s_correct = 0
    total = 0
    loss = 0
    best_loss = best_loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    test_image_path = args.fea_test_dir + '/images'
    test_label_path = args.fea_test_dir + '/labels'
    test_dataset = ImageTrainDataSet(test_image_path, test_label_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        print(">>>>>>>>>>>>>>>>>>>>>> feature testing time <<<<<<<<<<<<<<<<<<<<<<<<")
        for images, labels in tqdm(test_dataloader):
            images, labels = images.to(device), labels.to(device)

            labels[:,0][labels[:,0] == 999] = 0
            labels[:,1][labels[:,1] == 999] = 0
            labels[:,2][labels[:,2] == 999] = 0

            boundary_out, direction_out, shape_out= model(images)


            boundary_loss = criterion(boundary_out, labels[:,0])
            direction_loss = criterion(direction_out, labels[:, 1])
            shape_loss = criterion(shape_out, labels[:, 2])
            loss = boundary_loss + direction_loss + shape_loss

            running_loss += loss.item()

            _, b_predicted = torch.max(boundary_out, 1)
            b_correct += (b_predicted == labels[:, 0]).sum().item()

            _, d_predicted = torch.max(direction_out, 1)
            d_correct += (d_predicted == labels[:, 1]).sum().item()

            _, s_predicted = torch.max(shape_out, 1)
            s_correct += (s_predicted == labels[:, 2]).sum().item()
            
            total += images.size(0)
         
        epoch_loss = running_loss / len(test_dataloader)
        if not os.path.exists(args.save_path):
              os.makedirs(args.save_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), args.save_path + '/best_model.pth')
            print(f"Best Model saved with loss {best_loss:.4f}")
        
        b_accuracy = 100 * b_correct / total
        d_accuracy = 100 * d_correct / total
        s_accuracy = 100 * s_correct / total
        print(f'Test Loss: {epoch_loss:.4f}, Boundary Test Accuracy: {b_accuracy:.2f}%')
        print(f'Test Loss: {epoch_loss:.4f}, Direction Test Accuracy: {d_accuracy:.2f}%')
        print(f'Test Loss: {epoch_loss:.4f}, Shape Test Accuracy: {s_accuracy:.2f}%')
        

        return best_loss