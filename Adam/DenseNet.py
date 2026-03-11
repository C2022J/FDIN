import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os

# Use a non-interactive backend for Matplotlib
import matplotlib
from torchvision.models import DenseNet121_Weights

matplotlib.use('Agg')

# Set hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 20

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Store loss and accuracy
train_losses = {'Adam': [], 'SGD': []}
test_accuracies = {'Adam': [], 'SGD': []}

# Training and evaluation function
def train_and_evaluate(optimizer_name):
    # Load DenseNet model each time before training
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 10)  # Modify final layer for 10 classes
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

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(f"{optimizer_name} Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_losses[optimizer_name].append(epoch_loss / len(train_loader))
        print(f"{optimizer_name} Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {epoch_loss / len(train_loader):.4f}")

        # Test evaluation
        model.eval()
        correct, total = 0, 0
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
    os.makedirs('./data', exist_ok=True)

    # Optimizer settings
    optimizers = {
        'Adam': lambda params, lr: optim.Adam(params, lr=lr),
        'SGD': lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9)
    }

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train using both Adam and SGD optimizers
    train_and_evaluate('Adam')
    train_and_evaluate('SGD')

    # Plot and save training loss curve
    plt.figure(figsize=(6, 5))
    plt.plot(train_losses['Adam'], label='Adam')
    plt.plot(train_losses['SGD'], label='SGD')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/densenet_training_loss_comparison.png')

    # Plot and save test_V2 accuracy curve
    plt.figure(figsize=(6, 5))
    plt.plot(test_accuracies['Adam'], label='Adam')
    plt.plot(test_accuracies['SGD'], label='SGD')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/densenet_test_accuracy_comparison.png')
