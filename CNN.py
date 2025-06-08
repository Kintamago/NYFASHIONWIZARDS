import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
full_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# CNN with flexible activation
class SimpleDeepCNN(nn.Module):
    def __init__(self, activation_fn):
        super(SimpleDeepCNN, self).__init__()
        act = activation_fn()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), act,
            nn.Conv2d(32, 64, 3, padding=1), act,
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), act,
            nn.Conv2d(128, 128, 3, padding=1), act,
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), act,
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 128), act,
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# Train & Evaluate
def train_and_test(sample_size, activation_fn):
    train_subset, _ = torch.utils.data.random_split(full_trainset, [sample_size, len(full_trainset) - sample_size])
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

    model = SimpleDeepCNN(activation_fn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(1):  # 1 epoch
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    error = 100 * (1 - correct / total)
    return error

# Settings
learning_rates = [0.0001, 0.001, 0.005, 0.01, 0.05]
sample_size = 30000
activation_fn = nn.ReLU  # fixed activation

errors = []

for lr in learning_rates:
    def train_with_lr(sample_size, activation_fn, lr):
        train_subset, _ = torch.utils.data.random_split(full_trainset, [sample_size, len(full_trainset) - sample_size])
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

        model = SimpleDeepCNN(activation_fn).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for _ in range(1):  # 1 epoch
            for imgs, labels in trainloader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(imgs), labels)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return 100 * (1 - correct / total)

    error = train_with_lr(sample_size, activation_fn, lr)
    errors.append(error)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, errors, marker='o')
plt.xscale('log')
plt.title("Test Error vs Learning Rate")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Test Error Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("lr_vs_error.png")
plt.show()
