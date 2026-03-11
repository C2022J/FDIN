import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
from torchvision.models import ResNet50_Weights

# 使用非交互式后端
import matplotlib

matplotlib.use('Agg')

# 设置超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 20

# 数据加载和预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 存储损失和准确度
train_losses = {'Adam': [], 'SGD': []}
test_accuracies = {'Adam': [], 'SGD': []}


# 训练与评估函数
def train_and_evaluate(optimizer_name):
    # 每次训练前重新加载ResNet50模型
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)  # 修改最后一层为10个类别
    model = model.to(device)

    optimizer = optimizers[optimizer_name](model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 打印每个批次的损失
            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(
                    f"{optimizer_name} Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_losses[optimizer_name].append(epoch_loss / len(train_loader))
        print(
            f"{optimizer_name} Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {epoch_loss / len(train_loader):.4f}")

        # 在测试集上评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies[optimizer_name].append(accuracy)
        print(f"{optimizer_name} Test Accuracy after Epoch [{epoch + 1}/{num_epochs}]: {accuracy:.2f}%\n")
        model.train()


if __name__ == '__main__':
    # 确保保存路径存在
    os.makedirs('./data', exist_ok=True)

    # 优化器设置
    optimizers = {
        'Adam': lambda params, lr: optim.Adam(params, lr=lr),
        'SGD': lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9)
    }

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 对Adam和SGD分别进行训练
    train_and_evaluate('Adam')
    train_and_evaluate('SGD')

    # 绘制并保存训练损失曲线
    plt.figure(figsize=(6, 5))
    plt.plot(train_losses['Adam'], label='Adam')
    plt.plot(train_losses['SGD'], label='SGD')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/training_loss_comparison.png')

    # 绘制并保存验证集准确度曲线
    plt.figure(figsize=(6, 5))
    plt.plot(test_accuracies['Adam'], label='Adam')
    plt.plot(test_accuracies['SGD'], label='SGD')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/test_accuracy_comparison.png')
