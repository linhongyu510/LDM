```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

# 1. 定义超参数
# 批次大小，即每次训练时输入模型的样本数量，较大的批次可以利用GPU并行计算能力，但可能占用更多内存
BATCH_SIZE = 32
# 训练轮数，模型对整个训练数据集进行完整训练的次数
EPOCHS = 30
# 学习率，控制模型在训练过程中参数更新的步长大小，影响模型的收敛速度和最终性能
LEARNING_RATE = 0.001
# 类别数量，这里针对AID数据集设置为30类，用于最后分类层的输出维度
NUM_CLASSES = 30  

# 2. 数据预处理
# Compose函数用于将多个数据变换操作组合在一起，按顺序依次对数据进行处理
transform = transforms.Compose([
    # 将输入图像调整为224x224的尺寸，这是GoogLeNet模型要求的输入图像大小
    transforms.Resize((224, 224)),  
    # 将PIL图像或numpy.ndarray类型的数据转换为PyTorch的张量，同时将像素值从[0, 255]缩放到[0, 1]
    transforms.ToTensor(),
    # 对图像进行归一化处理，根据ImageNet数据集的统计信息设置均值和标准差，有助于模型更快收敛和提高性能
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# 3. 加载AID数据集 (假设路径为"./AID")
# ImageFolder类用于从指定路径加载图像数据集，每个子文件夹对应一个类别，transform参数指定对加载图像进行的预处理操作
dataset = datasets.ImageFolder(root="./AID", transform=transform)

# 划分训练集和验证集（80%训练集，20%验证集）
# 计算训练集的样本数量，占总数据集的80%
train_size = int(0.8 * len(dataset))
# 计算验证集的样本数量，总样本数减去训练集样本数
val_size = len(dataset) - train_size
# 使用random_split函数随机划分数据集为训练集和验证集，比例为[train_size, val_size]
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建训练集的数据加载器，设置批次大小、是否打乱数据和工作进程数
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# 创建验证集的数据加载器，设置批次大小、是否打乱数据和工作进程数，验证集一般不打乱数据
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 4. 定义GoogLeNet模型
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        # 是否使用辅助分类器，GoogLeNet中用于辅助训练的结构
        self.aux_logits = aux_logits

        # 第一个卷积层，输入通道为3（RGB图像），输出通道为64，卷积核大小为7，步长为2，填充为3
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # 第一个最大池化层，池化核大小为3，步长为2，ceil_mode=True表示向上取整
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 第二个卷积层，输入通道为64，输出通道为64，卷积核大小为1
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        # 第三个卷积层，输入通道为64，输出通道为192，卷积核大小为3，填充为1
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        # 第二个最大池化层，池化核大小为3，步长为2，ceil_mode=True
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 第一个Inception模块，定义了不同分支的通道数配置
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # 第二个Inception模块
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # 第三个最大池化层，池化核大小为3，步长为2，ceil_mode=True
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 第三个Inception模块
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # 第四个Inception模块
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # 第五个Inception模块
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # 第六个Inception模块
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # 第七个Inception模块
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        # 第四个最大池化层，池化核大小为3，步长为2，ceil_mode=True
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 第八个Inception模块
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        # 第九个Inception模块
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 如果使用辅助分类器，则定义两个辅助分类器模块
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        # 自适应平均池化层，将输出特征图调整为1x1的大小
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout层，防止过拟合，以一定概率（这里是0.4）随机将神经元输出置为0
        self.dropout = nn.Dropout(0.4)
        # 全连接层，将池化后的特征映射到指定的类别数量（num_classes）
        self.fc = nn.Linear(1024, num_classes)
        # 如果需要初始化权重，则调用初始化函数
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        # 如果是训练模式且使用辅助分类器，则计算第一个辅助分类器的输出
        if self.training and self.aux_logits:    
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        # 如果是训练模式且使用辅助分类器，则计算第二个辅助分类器的输出
        if self.training and self.aux_logits:    
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        # 如果是训练模式且使用辅助分类器，则返回主输出和两个辅助分类器的输出
        if self.training and self.aux_logits:   
            return x, aux2, aux1
        # 否则只返回主输出
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化方法初始化卷积层权重，mode='fan_out'表示使用fan_out模式，nonlinearity='relu'表示激活函数为ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 将卷积层的偏置初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 将全连接层的权重初始化为均值为0，标准差为0.01的正态分布
                nn.init.normal_(m.weight, 0, 0.01)
                # 将全连接层的偏置初始化为0
                nn.init.constant_(m.bias, 0)

# 5. 定义辅助模块和基本卷积层
# Inception模块，是GoogLeNet的核心结构，包含多个分支，不同分支使用不同的卷积核
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # 第一个分支，1x1卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # 第二个分支，先1x1卷积降维，再3x3卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 第三个分支，先1x1卷积降维，再5x5卷积
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # 第四个分支，先最大池化，再1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        # 在通道维度上拼接四个分支的输出
        return torch.cat(outputs, 1)


# 辅助分类器模块，用于辅助训练，提高模型的收敛性
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        # 平均池化层，池化核大小为5，步长为3
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        # 1x1卷积层，将输入通道数转换为128
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        # 第一个全连接层，输入维度为2048，输出维度为1024
        self.fc1 = nn.Linear(2048, 1024)
        # 第二个全连接层，输入维度为1024，输出维度为类别数量（num_classes）
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        # Dropout层，防止过拟合，以0.5的概率随机将神经元输出置为0
        x = F.dropout(x, 0.5, training=self.training)
        # ReLU激活函数，inplace=True表示直接在原张量上进行操作，节省内存
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


# 基本卷积层模块，包含一个卷积层和一个ReLU激活层
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        # 卷积层，根据传入的参数（如kernel_size等）创建卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        # ReLU激活层，inplace=True表示直接在原张量上进行操作，节省内存
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# 6. 选择训练设备
# 判断是否有可用的GPU，如果有则使用GPU进行训练，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建GoogLeNet模型实例，设置类别数量，并将模型移动到指定设备上（GPU或CPU）
model = GoogLeNet(num_classes=NUM_CLASSES).to(device)

# 7. 定义损失函数和优化器
# 使用交叉熵损失函数，适用于多分类任务
criterion = nn.CrossEntropyLoss()
# 使用Adam优化器，用于更新模型的参数，传入模型的可学习参数和学习率
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 8. 训练模型
def train_model():
    for epoch in range(EPOCHS):
        # 将模型设置为训练模式，启用Dropout、Batch Normalization等在训练时的操作
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # 将输入数据和标签数据移动到指定的设备（GPU或CPU）上
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零，防止梯度累加影响参数更新
            optimizer.zero_grad()
            # 前向传播，通过模型计算输出
            outputs = model(inputs)
            # 计算损失值，根据模型输出和真实标签计算交叉熵损失
            loss = criterion(outputs, labels)
            # 反向传播，计算梯度
            loss.backward()
            # 根据计算得到的梯度更新模型参数
            optimizer.step()

            running_loss += loss.item()
            # 获取每个样本预测的类别索引，返回值中第一个是最大值，第二个是对应的索引，这里只取索引
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
# 保存模型的参数，而不是整个模型结构，这样可以减小文件大小，保存的参数可以在后续用于加载模型进行推理或继续训练
torch.save(model.state_dict(), "googlenet_aid.pth")
```

