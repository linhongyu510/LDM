```python
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
    def __init__(self, num_classes=30):  # AID 有 30 个类别 224*224
        super(LeNet, self).__init__()		
       
        self.conv1 = nn.Conv2d(3, 16, 5)		# 卷积层：输入通道3 输出通道16 卷积核大小5*5 (224-5+2*0)/1+1=220
        self.pool1 = nn.MaxPool2d(2, 2)			# 池化层：池化 (220-2)/2+1=110
	        
        self.conv2 = nn.Conv2d(16, 32, 5)   # 卷积层：输入通道16 输出通道32 卷积核大小5*5 (110-5+2*0)/2+1=106
        self.pool2 = nn.MaxPool2d(2, 2)			# 池化层：池化 (106-2)/2+1=53
        
        self.fc1 = nn.Linear(32*53*53, 120)  # 输入尺寸修改为 224x224 W:53 H:53 输入通道：32 总32*53*53
        self.fc2 = nn.Linear(120, 84)        # 输入通道：120 输出通道：84
        self.fc3 = nn.Linear(84, num_classes)  # 30 类 输入通道：84 输出通道：num_classes
    
    def forward(self, x):
        x = self.conv1(x) # 卷积层
        x = F.relu(x) 		# 激活函数层
        x = self.pool1(x) # 池化层
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(-1, 32*53*53)  # 根据 AID 数据集输入尺寸计算展平后大小 展平
        x = self.fc1(x)		# 全连接层
        x = F.relu(x)			# 激活函数层
        
        x = self.fc2(x)		# 全连接层
        x = F.relu(x)			# 激活函数层
        x = self.fc3(x)		# 全连接层
        return x

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')

# AID 数据集路径
data_root = os.path.abspath("./AID")  # 你的 AID 数据集路径
train_dir = data_root  # AID 目录本身就是分类文件夹

# 预处理 transforms.Compose()将多个数据增强函数集合在一起，如RandomResizeCrop, RandomHorizontalClip, ColorJitter, RandomRotation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一输入尺寸
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.ToTensor(), # 转换成张量的形式
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 正则化 在PyTorch中，transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 是数据预处理中的归一化操作，用于对图像数据进行标准化处理，使数据具有更好的分布特性，有助于模型训练的收敛和性能提升。减去均值，除以标准差，将数据映射到一个新的范围
])

# 加载数据集
dataset = datasets.ImageFolder(root=train_dir, transform=transform)

# 训练/验证集拆分
train_size = int(0.8 * len(dataset))  # 80% 训练集
val_size = len(dataset) - train_size  # 20% 验证集

# torch.utils.data.random_split函数用于将一个数据集 dataset 随机划分为两个子集，通常用于划分训练集和验证集
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 类别信息
class_names = dataset.classes
print(f"Classes: {class_names}")

# 初始化 LeNet
net = LeNet(num_classes=len(class_names)).to(device)

# 损失函数 & 优化器
loss_function = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001) # 优化器选择Adam，将网络全部参数net.parameters()和学习率lr传入

# 训练
epochs = 30 # 训练的轮数
best_acc = 0.0 # 记录最好的准确率
for epoch in range(epochs):
    net.train() # 训练模式
    running_loss = 0.0 # 记录训练的损失
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # 在深度学习训练过程中，当使用反向传播算法计算梯度时，每次计算得到的梯度是基于当前批次数据的。如果不将梯度清零，在进行下一次反向传播时，新计算的梯度会与之前保留的梯度累加起来。这会导致梯度计算错误，使得模型的更新方向出现偏差，最终影响模型的训练效果。
        optimizer.zero_grad() # 先将梯度清零，不然梯度会一直累计
        outputs = net(images) # outputs即为模型预测值
        loss = loss_function(outputs, labels) # 将预测值pred与标签ground_truth放入损失函数计算损失
        loss.backward() # 梯度回传
        optimizer.step() # 优化器优化模型
        running_loss += loss.item() # 将损失加入总的训练损失中
    
    # 计算验证集准确率
    net.eval() # 测试模式
    correct = 0 # 准确的数量
    total = 0 # 总的数量
    with torch.no_grad(): # 测试的时候不计算梯度，进入模型就会计算梯度，所以要with.no_grad()防止对模型进行修改
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images) # 得到模型的预测结果
            _, predicted = torch.max(outputs, 1)  # 得到类别最大值，1表示的是类别维度，0维度为batch_size
            total += labels.size(0)  # 总数量加1
            correct += (predicted == labels).sum().item() # 如果预测值与标签相等，则准确的数量加1
    
    val_acc = correct / total # 计算这个epoch的准确率并打印
    print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

    # 保存最优模型
    if val_acc > best_acc: # 当前训练的模型准确率比之前最好的高，则保存当前的模型文件为pth
        best_acc = val_acc
        torch.save(net.state_dict(), "LeNet_AID.pth")

print("Training Finished")
if __name__ == '__main__':
    main()


```
