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
# 4. Plot the results
# The plot is saved as 'kNN_accuracy_plot.png' and displayed using plt.show()
# The code above loads the FashionMNIST dataset, flattens the images, trains a kNN classifier with varying k values,
# and plots the training and testing accuracies against the number of neighbors.
# The plot shows how the accuracy changes with different values of k, helping to visualize the performance of the kNN classifier.
# The code uses sklearn's KNeighborsClassifier to perform the kNN classification.
# The training and testing accuracies are stored in lists and plotted using matplotlib.
