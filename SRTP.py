import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ----------------- 1. 数据预处理 -----------------
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/root/autodl-tmp/SRTP/OpenDataLab___AID/raw/AID/AID'
dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 分别为训练和验证设置不同的数据增强
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = dataset.classes

# ----------------- 2. 模型构建 -----------------
def build_model(model, num_classes):
    if isinstance(model, models.VGG):
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'fc'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Unsupported model architecture")
    return model

model_vgg = build_model(models.vgg16(pretrained=True), len(class_names))
model_inception = build_model(models.inception_v3(pretrained=True), len(class_names))
model_resnet = build_model(models.resnet50(pretrained=True), len(class_names))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------- 3. 模型训练与评估函数 -----------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    # 针对 Inception 模型，若输出不是 Tensor则提取 logits
                    if not isinstance(outputs, torch.Tensor):
                        outputs = outputs.logits
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase=='val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()
    print(f'Best Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def train_and_evaluate(model):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return train_model(model, criterion, optimizer, scheduler, num_epochs=30)

# 分别训练三个模型
model_vgg = train_and_evaluate(model_vgg)
model_inception = train_and_evaluate(model_inception)
model_resnet = train_and_evaluate(model_resnet)

def evaluate_single_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs.logits
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    return acc, cm, report

print("\n单模型评估结果:")
print("VGG-16:")
vgg_acc, vgg_cm, vgg_report = evaluate_single_model(model_vgg, dataloaders['val'])
print(f"Accuracy: {vgg_acc:.4f}\nConfusion Matrix:\n{vgg_cm}\nReport:\n{vgg_report}")

print("Inception-v3:")
incep_acc, incep_cm, incep_report = evaluate_single_model(model_inception, dataloaders['val'])
print(f"Accuracy: {incep_acc:.4f}\nConfusion Matrix:\n{incep_cm}\nReport:\n{incep_report}")

print("ResNet-50:")
resnet_acc, resnet_cm, resnet_report = evaluate_single_model(model_resnet, dataloaders['val'])
print(f"Accuracy: {resnet_acc:.4f}\nConfusion Matrix:\n{resnet_cm}\nReport:\n{resnet_report}")

# ----------------- 4. 特征融合 -----------------
# 对于VGG和ResNet，使用通用特征提取器；对Inception使用专门的特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        return self.features(x)

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(InceptionFeatureExtractor, self).__init__()
        self.model = model
    def forward(self, x):
        # 手动前向计算至avgpool层
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# 构造特征提取器列表
feature_extractors = []
feature_extractors.append(FeatureExtractor(model_vgg).to(device))
feature_extractors.append(InceptionFeatureExtractor(model_inception).to(device))
feature_extractors.append(FeatureExtractor(model_resnet).to(device))

def extract_features(models, dataloader):
    features, labels = [], []
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            model_outputs = []
            for model in models:
                out = model(inputs)
                model_outputs.append(out.view(inputs.size(0), -1))
            fused_features = torch.cat(model_outputs, dim=1)
            features.append(fused_features.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

train_features, train_labels = extract_features(feature_extractors, dataloaders['train'])
val_features, val_labels = extract_features(feature_extractors, dataloaders['val'])

# 使用 PCA 降维（降到500维）
pca = PCA(n_components=500)
train_features = pca.fit_transform(train_features)
val_features = pca.transform(val_features)

# 训练 Logistic 回归分类器
clf = LogisticRegression(max_iter=2000)
clf.fit(train_features, train_labels)
fusion_preds = clf.predict(val_features)
fusion_acc = accuracy_score(val_labels, fusion_preds)
print(f"\n特征融合后 Logistic 回归模型准确率: {fusion_acc:.4f}")

# ----------------- 5. 可视化示例 -----------------
# 如果有记录训练过程的loss和acc，可使用下面代码进行可视化（此处仅为示例）
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.show()
