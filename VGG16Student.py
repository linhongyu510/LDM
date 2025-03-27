import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split

# 1. 定义超参数
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.001
NUM_CLASSES = 30  # AID 数据集有 30 类

# 2. 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG 需要 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 3. 加载 AID 数据集 (假设路径为 "./AID")
dataset = datasets.ImageFolder(root="./AID", transform=transform)

# 划分训练集和验证集（80% 训练集，20% 验证集）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 4. 加载 VGG16 预训练模型
model = models.vgg16(pretrained=True)

# 5. 替换全连接层，使其适配 AID 数据集
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

# 6. 选择训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 7. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 8. 训练模型
def train_model():
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")

train_model()

# 9. 保存模型
torch.save(model.state_dict(), "vgg16_aid.pth")
