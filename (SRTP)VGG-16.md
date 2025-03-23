```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split

# 1. 定义超参数
# 批次大小，每次训练输入模型的样本数量
BATCH_SIZE = 32

# 训练轮数，即对整个训练数据集进行完整训练的次数
EPOCHS = 30

# 学习率，控制模型参数更新的步长大小
LEARNING_RATE = 0.001

# 类别数量，AID 数据集有 30 类
NUM_CLASSES = 30  

# 2. 数据预处理
# Compose 用于将多个数据变换操作组合在一起
transform = transforms.Compose([
    # 将图像调整为 224x224 的尺寸，这是 VGG16 模型所要求的输入图像大小
    transforms.Resize((224, 224)),  
    # 将 PIL 图像或 numpy.ndarray 转换为 PyTorch 的张量，并且将像素值从 [0, 255] 缩放到 [0, 1]
    transforms.ToTensor(),
    # 对图像进行归一化处理，根据 ImageNet 数据集的统计信息设置均值和标准差
  
    # 归一化有助于模型更快地收敛和提高性能
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# 3. 加载 AID 数据集 (假设路径为 "./AID")
# ImageFolder 用于从指定路径加载图像数据集，每个子文件夹对应一个类别
# transform 参数指定对加载的图像进行预处理操作
dataset = datasets.ImageFolder(root="./AID", transform=transform)

# 划分训练集和验证集（80% 训练集，20% 验证集）
# 计算训练集的样本数量
train_size = int(0.8 * len(dataset))

# 计算验证集的样本数量
val_size = len(dataset) - train_size

# 使用 random_split 函数随机划分数据集为训练集和验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建训练集的数据加载器，设置批次大小、是否打乱数据和工作进程数
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# 创建验证集的数据加载器，设置批次大小、是否打乱数据和工作进程数
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 4. 加载 VGG16 预训练模型
# models.vgg16(pretrained=True) 表示加载在 ImageNet 数据集上预训练好的 VGG16 模型
model = models.vgg16(pretrained=True)

# 5. 替换全连接层，使其适配 AID 数据集
# VGG16 原始的全连接层最后一层输出维度是为 ImageNet 的 1000 类设计的
# 这里我们需要将其修改为 AID 数据集的类别数（30 类）
# model.classifier[6] 表示全连接层中的最后一层，将其替换为新的线性层
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

# 6. 选择训练设备
# 判断是否有可用的 GPU，如果有则使用 GPU 进行训练，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移动到指定的设备（GPU 或 CPU）上
model = model.to(device)

# 7. 定义损失函数和优化器
# 使用交叉熵损失函数，适用于多分类任务
criterion = nn.CrossEntropyLoss()

# 使用 Adam 优化器，用于更新模型的参数
# 传入模型的可学习参数和学习率
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 8. 训练模型
def train_model():
    for epoch in range(EPOCHS):
        # 将模型设置为训练模式，启用 dropout、batch normalization 等在训练时的操作
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # 将输入数据和标签数据移动到指定的设备（GPU 或 CPU）上
            inputs, labels = inputs.to(device), labels.to(device)
            # 1.梯度清零，防止梯度累加影响参数更新
            optimizer.zero_grad()
            # 2.前向传播，通过模型计算输出
            outputs = model(inputs)
            # 3.计算损失值
            loss = criterion(outputs, labels)
            # 4.反向传播，计算梯度
            loss.backward()
            # 5.根据计算得到的梯度更新模型参数
            optimizer.step()

            running_loss += loss.item()
            # 获取每个样本预测的类别索引 第一个参数为batch_size 不需要所以用无意义变量名_
            _, predicted = outputs.max(1)
            
            # 统计预测正确的样本数量
            correct += (predicted == labels).sum().item()
            
            # 统计总样本数量
            total += labels.size(0)

        # 计算训练集上的准确率
        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")

# 调用训练函数开始训练模型
train_model()

# 9. 保存模型
# 保存模型的参数，而不是整个模型结构，这样可以减小文件大小
# 保存的参数可以在后续用于加载模型进行推理或继续训练
torch.save(model.state_dict(), "vgg16_aid.pth")
```

