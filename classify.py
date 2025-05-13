# 导入常用库
import torch                     # PyTorch 主库
import os                        # 操作文件路径用
from torchvision import datasets # 加载图像数据集用
import torchvision.transforms as transforms  # 图像预处理和增强用
import matplotlib                # 配置图像显示用
import matplotlib.pyplot as plt  # 绘图库
import json                      # 保存和读取 JSON 文件用

# 设置 matplotlib 字体，防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']        # 设置中文黑体
matplotlib.rcParams['axes.unicode_minus'] = False          # 防止负号显示成方块

# 定义图像增强与预处理方式
dataset_transform = transforms.Compose([
    transforms.RandomResizedCrop((128, 128)),  # 随机裁剪并调整为 128×128
    transforms.ToTensor(),                     # 转换为 PyTorch 张量，像素值范围 [0, 1]
    transforms.Normalize(                      # 标准化 RGB 三个通道，提升收敛速度
        (0.485, 0.456, 0.406),                 # 均值
        (0.229, 0.224, 0.225)                  # 方差
    )
])

# 指定数据集文件夹路径
dataset_dir = 'animal'
train_dir = os.path.join(dataset_dir, 'train')  # 训练集目录
test_dir = os.path.join(dataset_dir, 'val')     # 测试集目录

# 加载数据集
train_dataset = datasets.ImageFolder(train_dir, dataset_transform)
test_dataset = datasets.ImageFolder(test_dir, dataset_transform)

# 用 DataLoader 封装，方便批量读取
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 获取数据集中的分类标签名（目录名就是类别名）
classes = train_dataset.classes
print(f"类别: {classes}")

# 将类别标签保存到 JSON 文件，方便后续使用
with open('classes.json', 'w', encoding='utf-8') as f:
    json.dump(classes, f, ensure_ascii=False)

# 设备检测：优先用 GPU，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'当前设备：{device}')

# 导入神经网络相关模块
import torch.nn as nn
import torch.optim as optim

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # 定义卷积层，每个卷积层后接 ReLU 激活函数和池化层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)  # 输入3通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 最大池化

        # 自适应平均池化，把任意输入尺寸变成固定 4x4 尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)  # 输出类别数个神经元

    def forward(self, x):
        # 前向传播过程
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = self.avgpool(x)        # 保证无论输入多大，最后都是 4x4
        x = torch.flatten(x, 1)    # 展平成一维向量，dim=1 表示从第二维开始展平
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型，指定类别数，放到 GPU/CPU 上
model = CNN(num_classes=len(classes)).to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()  # 多分类常用损失函数

# 准备记录训练过程中的损失和准确率
train_losses = []
train_accuracies = []
test_accuracies = []

# 开始训练循环，跑 50 个 epoch
for epoch in range(50):
    model.train()  # 切换为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 放到 GPU/CPU
        optimizer.zero_grad()                  # 梯度清零
        outputs = model(images)                # 前向计算
        loss = loss_fn(outputs, labels)        # 计算损失
        loss.backward()                        # 反向传播
        optimizer.step()                       # 更新权重

        running_loss += loss.item()            # 累计损失
        _, predicted = torch.max(outputs.data, 1)  # 取每行最大值的下标
        total += labels.size(0)                # 累计总样本数
        # (predicted == labels) 是一个布尔张量 torch.BoolTensor，.sum() 是可以的，会把 True 当 1，False 当 0
        correct += (predicted == labels).int().sum().item()  # 累计预测正确数量

    # 计算本轮平均损失和准确率
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print(f'第 {epoch+1}/50 轮训练完成，损失：{train_loss:.4f}，准确率：{train_accuracy:.2f}%')

    # 模型评估模式
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 测试时不计算梯度，节省内存和加速
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).int().sum().item()

        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
    print(f'测试集准确率：{test_accuracy:.2f}%')

# 保存模型参数
torch.save(model.state_dict(), 'animal_classification_model.pth')

# 绘制训练损失和准确率曲线
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.title('训练损失')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='训练准确率')
plt.plot(test_accuracies, label='测试准确率')
plt.xlabel('训练轮次')
plt.ylabel('准确率 (%)')
plt.title('训练和测试准确率')
plt.legend()

plt.tight_layout()  # 自动调整子图间距
plt.savefig('train_loss_accuracy.png')  # 保存图像
plt.show()

# 保存损失和准确率数据到 JSON 文件
with open('train_loss_accuracy.json', 'w', encoding='utf-8') as f:
    json.dump({
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }, f, ensure_ascii=False)
