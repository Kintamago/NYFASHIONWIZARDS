import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

x_train = train_dataset.data.numpy().reshape(len(train_dataset), -1)
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy().reshape(len(test_dataset), -1)
y_test = test_dataset.targets.numpy()

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --- PART 1: Regularization Strength Analysis ---
cs = [0.01, 0.1, 1, 10, 50]
p = 'l1'  # penalty

train_accuracies = []
test_accuracies = []

for c in cs:
    model = LogisticRegression(
        penalty=p,
        max_iter=100,
        C=c,
        solver='saga',
        dual=False,
        verbose=0,
        random_state=69
    )

    model.fit(x_train_scaled, y_train)

    train_pred = model.predict(x_train_scaled)
    test_pred = model.predict(x_test_scaled)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"\nRegularization Strength (C): {c} and Penalty: {p}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

# --- PART 2: Varying Training Data Size ---
data_sizes = [10000, 20000, 30000, 40000, 50000, 60000]
size_train_accuracies = []
size_test_accuracies = []

best_c = 50
best_penalty = 'l2'

for size in data_sizes:
    indices = np.random.choice(len(x_train_scaled), size, replace=False)
    x_train_subset = x_train_scaled[indices]
    y_train_subset = y_train[indices]

    model = LogisticRegression(
        penalty=best_penalty,
        max_iter=100,
        C=best_c,
        solver='saga',
        dual=False,
        random_state=69
    )

    model.fit(x_train_subset, y_train_subset)

    train_pred = model.predict(x_train_subset)
    test_pred = model.predict(x_test_scaled)

    train_acc = accuracy_score(y_train_subset, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    size_train_accuracies.append(train_acc)
    size_test_accuracies.append(test_acc)

    print(f"\nTraining data size: {size}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

# --- COMBINED PLOTTING ---
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot for Regularization Strength
axs[0].plot(cs, train_accuracies, 'b-o', label='Training Accuracy')
axs[0].plot(cs, test_accuracies, 'r-o', label='Test Accuracy')
axs[0].set_xscale('log')
axs[0].set_xlabel('Regularization Strength (C)')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Accuracy vs Regularization Strength')
axs[0].legend()
axs[0].grid(True)

# Plot for Training Data Size
axs[1].plot(data_sizes, size_train_accuracies, 'b-o', label='Training Accuracy')
axs[1].plot(data_sizes, size_test_accuracies, 'r-o', label='Test Accuracy')
axs[1].set_xlabel('Training Data Size')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Accuracy vs Training Data Size')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("logistic_regression_analysis.png", dpi=300)  # Optional
plt.show()
