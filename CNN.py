import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create models directory first
os.makedirs('./models', exist_ok=True)

# Define Larger CNN
class BiggerCNN(nn.Module):
    def __init__(self):
        super(BiggerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 1 * 1, 512) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(torch.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = self.pool(torch.relu(self.conv3(x)))   # 7x7 -> 3x3
        x = self.pool(torch.relu(self.conv4(x)))   # 3x3 -> 1x1
        x = x.view(-1, 128 * 1 * 1) 
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
# Load Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True)

# Instantiate model, loss, optimizer
net = BiggerCNN().to(device)  # move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

print("Starting training...")

# Train Loop
for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)  # move data to GPU

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")

# Save the model
torch.save(net.state_dict(), './models/fashion_cnn_weights.pth')
print("Model weights saved to './models/fashion_cnn_weights.pth'")

torch.save(net, './models/fashion_cnn_complete.pth')
print("Complete model saved to './models/fashion_cnn_complete.pth'")
