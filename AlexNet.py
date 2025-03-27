import torch.nn as nn
import torch
import os
import sys
import json
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

class AlexNet(nn.Module):
    def __init__(self, num_classes=30, init_weights=False):  # AID 数据集有 30 类
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)  # 30 类
        )
        
        if init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using {} device.'.format(device))
    
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    
    # 你的 AID 数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    image_path = os.path.join(data_root, "SRTP", "AID")  # 确保路径正确
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 读取完整数据集
    full_dataset = datasets.ImageFolder(root=image_path, transform=data_transform["train"])
    
    # 80% 训练，20% 验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 训练集应用训练变换，验证集应用验证变换
    train_dataset.dataset.transform = data_transform["train"]
    val_dataset.dataset.transform = data_transform["val"]
    
    # 获取类别索引
    class_to_idx = full_dataset.class_to_idx
    json_str = json.dumps(class_to_idx, indent=4)
    
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    batch_size = 32
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(num_workers))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validate_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=num_workers)

    print("Using {} images for training, {} images for validation.".format(train_size, val_size))
    
    net = AlexNet(num_classes=30, init_weights=True)  # AID 30 类
    net.to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    
    epochs = 30
    save_path = './AlexNet_AID.pth'
    best_acc = 0.0
    train_steps = len(train_loader)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.desc = "Train Epoch [{}/{}] Loss: {:.4f}".format(epoch + 1, epochs, loss.item())

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accurate = acc / val_size
        print('[Epoch %d] Train Loss: %.4f, Val Accuracy: %.4f' % (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print("Model Saved!")

    print('Finished Training')

if __name__ == '__main__':
    main()
