```python
import torch.nn as nn
import torch
import os
import sys
import json
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

# 定义AlexNet模型类
class AlexNet(nn.Module):
    def __init__(self, num_classes=30, init_weights=False):  
        # 调用父类的构造函数
        super(AlexNet, self).__init__()
        # 定义特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积层：输入通道3，输出通道48，卷积核大小11，步长4，填充2			(224-11+2*2)/4+1=55
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # ReLU激活函数，inplace=True表示直接在原张量上进行操作，节省内存
            nn.ReLU(inplace=True),
            # 最大池化层：池化核大小3，步长2
            nn.MaxPool2d(kernel_size=3, stride=2),		#(55-3)/2+1=27

            # 第二个卷积层：输入通道48，输出通道128，卷积核大小5，填充2					(27-5+2*2)/1+1=27
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),										# (27-3)/2+1=13

            # 第三个卷积层：输入通道128，输出通道192，卷积核大小3，填充1 	    (13-3+2*1)/1+1=13
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第四个卷积层：输入通道192，输出通道192，卷积核大小3，填充1       (13-3+2*1)/1+1=13
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第五个卷积层：输入通道192，输出通道128，卷积核大小3，填充1       (13-3+2*1)/1+1=13
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                     # (13-3)/2+1=6
        )

        # 定义分类器部分
        self.classifier = nn.Sequential(
            # Dropout层，防止过拟合，p=0.5表示随机将50%的神经元输出置为0
            nn.Dropout(p=0.5),
            # 全连接层：输入维度128*6*6，输出维度2048					in_channel:128 W:6 H:6
            nn.Linear(128*6*6, 2048), 
            nn.ReLU(inplace=True), # inplace=True表示直接在原张量上进行操作，节省内存
            # 全连接层：输入维度2048，输出维度2048
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # 全连接层：输入维度2048，输出维度为类别数（AID数据集为30类）
            nn.Linear(2048, num_classes)   # 最后一层的分类层输出通道数为目标数据集的类别数，即将结果映射到对应的类别
        )

        # 如果需要初始化权重
        if init_weights:
            self._initialize_weights()

    # 前向传播函数 提取特征，展平，进行分类
    def forward(self, x):
        x = self.features(x)
        # 将特征图展平，从第1个维度开始展平
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    # 权重初始化函数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):   # 如果是卷积层，使用Kaiming初始化卷积层权重
                # 使用Kaiming初始化方法初始化卷积层权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 将卷积层偏置初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 如果是全连接层，初始化为均值为0，方差为1的正态分布
                # 初始化全连接层权重为均值0，标准差0.01的正态分布
                nn.init.normal_(m.weight, 0, 0.01)
                # 将全连接层偏置初始化为0
                nn.init.constant_(m.bias, 0)

# 主函数
def main():
    # 确定使用的设备（GPU或CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using {} device.'.format(device))

    # 定义数据变换，训练集和验证集不同
    data_transform = {
        "train": transforms.Compose([			# 如果是训练集，则RandomResizeCrop, RandomHorizontalFlip, ColorJitter, RandomRotation
            # 随机裁剪并调整大小为224*224
            transforms.RandomResizedCrop(224),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
            # 将图像转换为张量
            transforms.ToTensor(),
            # 归一化，均值为(0.5, 0.5, 0.5)，标准差为(0.5, 0.5, 0.5)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([			 # 如果是验证集，则只需调整大小和进行数据归一化
            # 调整大小为224*224
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # 获取数据集根路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    
    # AID数据集路径
    image_path = os.path.join(data_root, "SRTP", "AID")  
    
    # 确保数据集路径存在
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 读取完整数据集
    full_dataset = datasets.ImageFolder(root=image_path, transform=data_transform["train"])

    # 划分数据集，80%用于训练，20%用于验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 分别设置训练集和验证集的变换
    train_dataset.dataset.transform = data_transform["train"]
    val_dataset.dataset.transform = data_transform["val"]

    # 获取类别索引
    class_to_idx = full_dataset.class_to_idx
    
    # 将类别索引转换为JSON字符串
    json_str = json.dumps(class_to_idx, indent=4)

    # 将类别索引保存为JSON文件
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 批次大小
    batch_size = 32
    
    # 数据加载器的工作进程数，取CPU核心数、批次大小（如果大于1）和8中的最小值
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(num_workers))

    # 创建训练集数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # 创建验证集数据加载器
    validate_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=num_workers)

    print("Using {} images for training, {} images for validation.".format(train_size, val_size))

    # 实例化AlexNet模型，设置类别数为30并初始化权重
    net = AlexNet(num_classes=30, init_weights=True)  
    
    # 将模型移动到指定设备上（GPU或CPU）
    net.to(device)

    # 定义损失函数为交叉熵损失
    loss_function = nn.CrossEntropyLoss()
    
    # 定义优化器为Adam优化器，学习率为0.0002 将模型的参net.parameters()和学习率lr传入
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    # 训练轮数
    epochs = 30
    
    # 模型保存路径
    save_path = './AlexNet_AID.pth'
    
    # 记录最佳验证准确率
    best_acc = 0.0
    
    # 训练步数，即训练集的批次数量
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # 设置模型为训练模式
        net.train()
        running_loss = 0.0
        
        # 使用tqdm显示训练进度条
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            
            # 将图像和标签移动到指定设备上
            images, labels = images.to(device), labels.to(device)

            # 1.梯度清零
            optimizer.zero_grad()
            
            # 2.前向传播
            outputs = net(images)
            
            # 3.计算损失
            loss = loss_function(outputs, labels)
            
            # 4.反向传播计算梯度
            loss.backward()
            
            # 5.更新模型参数
            optimizer.step()

            running_loss += loss.item()
            # 更新进度条描述信息
            train_bar.desc = "Train Epoch [{}/{}] Loss: {:.4f}".format(epoch + 1, epochs, loss.item())

        # 设置模型为评估模式
        net.eval()
        acc = 0.0
        # 不计算梯度，节省内存
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                outputs = net(val_images)
                # 获取预测的类别索引
                predict_y = torch.max(outputs, dim=1)[1]
                
                # 计算预测正确的数量
                acc += torch.eq(predict_y, val_labels).sum().item()

        # 计算验证集准确率
        val_accurate = acc / val_size
        print('[Epoch %d] Train Loss: %.4f, Val Accuracy: %.4f' % (epoch + 1, running_loss / train_steps, val_accurate))

        # 如果验证准确率高于当前最佳准确率，则保存模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print("Model Saved!")

    print('Finished Training')

if __name__ == '__main__':
    main()
```

