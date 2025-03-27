import torch.nn as nn
import torch
import os
import sys
import json
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import torch.nn.functional as F


# 定义教师网络 AlexNet
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
            nn.Linear(128 * 6 * 6, 2048),
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


# 定义学生网络（简单的卷积神经网络）
class StudentNet(nn.Module):
    def __init__(self, num_classes=30):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


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

    # 加载教师网络并设置为评估模式
    teacher_net = AlexNet(num_classes=30, init_weights=True)
    teacher_net.load_state_dict(torch.load('./AlexNet_AID.pth'))
    teacher_net.to(device)
    teacher_net.eval()

    # 初始化学生网络
    student_net = StudentNet(num_classes=30)
    student_net.to(device)

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    student_optimizer = optim.Adam(student_net.parameters(), lr=0.0002)

    # 知识蒸馏相关参数
    temperature = 10  # 温度参数
    alpha = 0.7  # 软目标损失权重

    epochs = 300
    save_path = './StudentNet_AID.pth'
    best_acc = 0.0
    train_steps = len(train_loader)

    for epoch in range(epochs):
        student_net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            student_optimizer.zero_grad()

            # 教师网络输出
            with torch.no_grad():
                teacher_outputs = teacher_net(images)
                soft_targets = F.softmax(teacher_outputs / temperature, dim=1)

            # 学生网络输出
            student_outputs = student_net(images)

            # 计算软目标损失（KL散度）
            soft_loss = F.kl_div(F.log_softmax(student_outputs / temperature, dim=1), soft_targets, reduction='batchmean')

            # 计算硬目标损失（交叉熵损失）
            hard_loss = loss_function(student_outputs, labels)

            # 总损失
            total_loss = alpha * soft_loss * (temperature ** 2) + (1 - alpha) * hard_loss

            total_loss.backward()
            student_optimizer.step()

            running_loss += total_loss.item()
            train_bar.desc = "Train Epoch [{}/{}] Loss: {:.4f}".format(epoch + 1, epochs, total_loss.item())

        student_net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                outputs = student_net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accurate = acc / val_size
        print('[Epoch %d] Train Loss: %.4f, Val Accuracy: %.4f' % (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(student_net.state_dict(), save_path)
            print("Model Saved!")

    print('Finished Training')


if __name__ == '__main__':
    main()