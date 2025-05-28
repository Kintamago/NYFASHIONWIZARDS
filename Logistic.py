import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the training data
train_dataset = datasets.FashionMNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

# Load the test data
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

x_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Changing Regularization Strength
cs = [0.01,
    #   0.1,
    #   1,
    #   10,
    #   50
      ]
penalties = ['l2',
            #   'l1'
            ]
train_accuracies = []
test_accuracies = []

for p in penalties:
    for c in cs:
        model = LogisticRegression(
            penalty=p,
            max_iter=5000,
            C=c,
            solver='liblinear',
            dual=False,
            random_state=69
        )

        model.fit(x_train_scaled, y_train)
        
        # Calculate accuracies
        train_pred = model.predict(x_train_scaled)
        test_pred = model.predict(x_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"\nRegularization Strength (C): {c} and Penalty: {p}")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(cs, train_accuracies, 'b-', label='Training Accuracy')
plt.plot(cs, test_accuracies, 'r-', label='Test Accuracy')
plt.xscale('log')  # Use log scale for better visualization
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy vs Regularization Strength')
plt.legend()
plt.grid(True)
plt.show()
    
