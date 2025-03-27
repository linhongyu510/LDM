import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os

# 修改 LeNet 适配 AID 数据集
class LeNet(nn.Module):
    def __init__(self, num_classes=30):  # AID 有 30 个类别
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32*53*53, 120)  # 输入尺寸修改为 224x224
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # 30 类

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(-1, 32*53*53)  # 根据 AID 数据集输入尺寸计算展平后大小
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')

    # AID 数据集路径
    data_root = os.path.abspath("./AID")  # 你的 AID 数据集路径
    train_dir = data_root  # AID 目录本身就是分类文件夹

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 统一输入尺寸
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    # 训练/验证集拆分
    train_size = int(0.8 * len(dataset))  # 80% 训练集
    val_size = len(dataset) - train_size  # 20% 验证集
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 类别信息
    class_names = dataset.classes
    print(f"Classes: {class_names}")

    # 初始化 LeNet
    net = LeNet(num_classes=len(class_names)).to(device)

    # 损失函数 & 优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 训练
    epochs = 30
    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 计算验证集准确率
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), "LeNet_AID.pth")

    print("Training Finished")

if __name__ == '__main__':
    main()
