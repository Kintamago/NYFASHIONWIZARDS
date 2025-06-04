import numpy as np
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load FashionMNIST (flattened tensors)
transform = transforms.Compose([
    transforms.ToTensor(),  # converts to shape (1, 28, 28)
])

train_set = datasets.FashionMNIST(root='.', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='.', train=False, download=True, transform=transform)

# 2. Extract images and labels, flatten the images to vectors
X_train = train_set.data.view(-1, 28 * 28).numpy() / 255.0  # normalize to [0,1]
y_train = train_set.targets.numpy()

X_test = test_set.data.view(-1, 28 * 28).numpy() / 255.0
y_test = test_set.targets.numpy()

# 3. Fit kNN classifier
n_sizes = [1, 3, 5, 7, 9, 11, 13, 15]
training_accuracies = []
testing_accuracies = []
for n in n_sizes:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    
    # Training accuracy
    y_train_pred = knn.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    training_accuracies.append(train_acc)
    
    # Testing accuracy
    y_test_pred = knn.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    testing_accuracies.append(test_acc)


plt.plot(n_sizes, training_accuracies, label='Training Accuracy', marker='o')
plt.plot(n_sizes, testing_accuracies, label='Testing Accuracy', marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('kNN Classifier Accuracy on FashionMNIST')
plt.xticks(n_sizes)
plt.legend()
plt.grid()
plt.savefig('kNN_accuracy_plot.png')
plt.show()

# Analysis with varying training data sizes
print("\nAnalyzing KNN performance with varying training data sizes...")
data_sizes = [10000, 20000, 30000, 40000, 50000, 60000]
size_train_accuracies = []
size_test_accuracies = []
size_train_errors = []
size_test_errors = []

# Use the best k value from previous analysis
best_k = 5  # You can adjust this based on your previous results

for size in data_sizes:
    # Create a subset of the training data
    indices = np.random.choice(len(X_train), size, replace=False)
    x_train_subset = X_train[indices]
    y_train_subset = y_train[indices]
    
    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train_subset, y_train_subset)
    
    # Calculate accuracies
    train_pred = knn.predict(x_train_subset)
    test_pred = knn.predict(X_test)
    
    train_acc = accuracy_score(y_train_subset, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    size_train_accuracies.append(train_acc)
    size_test_accuracies.append(test_acc)
    size_train_errors.append(1 - train_acc)
    size_test_errors.append(1 - test_acc)
    
    print(f"\nTraining data size: {size}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

# Plot the results for data size analysis
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, size_train_accuracies, 'b-', label='Training Accuracy')
plt.plot(data_sizes, size_test_accuracies, 'r-', label='Test Accuracy')
plt.plot(data_sizes, size_train_errors, 'b--', label='Training Error')
plt.plot(data_sizes, size_test_errors, 'r--', label='Test Error')
plt.xlabel('Training Data Size')
plt.ylabel('Rate')
plt.title(f'KNN Performance vs Training Data Size (k={best_k})')
plt.legend()
plt.grid(True)
plt.savefig('kNN_training_size_plot.png')
plt.show()

# The code above loads the FashionMNIST dataset, flattens the images, trains a kNN classifier with varying k values,
# and plots the training and testing accuracies against the number of neighbors.
# The plot shows how the accuracy changes with different values of k, helping to visualize the performance of the kNN classifier.
# The code uses sklearn's KNeighborsClassifier to perform the kNN classification.
# The training and testing accuracies are stored in lists and plotted using matplotlib.
