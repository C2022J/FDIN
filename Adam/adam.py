import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 数据集准备
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)


# 定义LeNet-5模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 训练函数
def train(model, optimizer, criterion, loader, epochs=5):
    model.train()
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(loader)
        losses.append(average_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')
    return losses


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型和损失函数
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()

# 定义不同的优化器
optimizers = {
    "Adam": optim.Adam(model.parameters(), lr=3e-4),
    "SGDNesterov": optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True),
    "AdaGrad": optim.Adagrad(model.parameters(), lr=0.01),
    "RMSProp": optim.RMSprop(model.parameters(), lr=0.01),
    "AdaDelta": optim.Adadelta(model.parameters(), lr=1.0),
}

# 训练并记录损失
losses_dict = {}
for opt_name, optimizer in optimizers.items():
    print(f'Training with {opt_name}...')
    model = LeNet5().to(device)  # 每次训练都重新初始化模型
    losses = train(model, optimizer, criterion, train_loader, epochs=20)
    losses_dict[opt_name] = losses

# 绘制损失曲线
plt.figure(figsize=(10, 6))
for opt_name, losses in losses_dict.items():
    plt.plot(losses, label=opt_name)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.yscale('log')  # 使用对数尺度更容易比较下降速度
plt.title('Optimizer Comparison on MNIST using LeNet-5')
plt.legend()
plt.grid(True)
plt.show()
