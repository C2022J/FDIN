import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig
import matplotlib.pyplot as plt
import os

# 设置超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# 数据加载和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载ViT模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=10
).to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 优化器设置
optimizers = {
    'Adam': optim.Adam(model.parameters(), lr=learning_rate),
    'SGD': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
}

# 存储损失和准确度
train_losses = {'Adam': [], 'SGD': []}
test_accuracies = {'Adam': [], 'SGD': []}

# 训练与评估函数
def train_and_evaluate(optimizer_name):
    optimizer = optimizers[optimizer_name]
    print(f"Starting training with {optimizer_name} optimizer...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # 输出每100个batch的损失
            if batch_idx % 100 == 0:
                print(f"{optimizer_name} Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        train_losses[optimizer_name].append(avg_loss)
        print(f"{optimizer_name} Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")

        # 测试集上评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies[optimizer_name].append(accuracy)
        print(f"{optimizer_name} Test Accuracy after Epoch [{epoch + 1}/{num_epochs}]: {accuracy:.2f}%\n")

if __name__ == '__main__':
    # 创建保存目录
    os.makedirs('./data', exist_ok=True)

    # 对Adam和SGD分别进行训练
    train_and_evaluate('Adam')
    train_and_evaluate('SGD')

    # 绘制训练损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses['Adam'], label='Adam')
    plt.plot(train_losses['SGD'], label='SGD')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/vit_training_loss_comparison_vit.png')

    # 绘制验证集准确度曲线
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies['Adam'], label='Adam')
    plt.plot(test_accuracies['SGD'], label='SGD')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/vit_test_accuracy_comparison_vit.png')

    # 显示图像
    plt.show()
