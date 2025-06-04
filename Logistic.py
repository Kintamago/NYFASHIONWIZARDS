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
# cs = [0.01,
#       0.1,
#       1,
#       10,
#       50
#       ]
# penalties = ['l1',
#             'l2'
#             ]
# train_accuracies = []
# test_accuracies = []

# for p in penalties:
#     for c in cs:
#         model = LogisticRegression(
#             penalty=p,
#             max_iter=500,
#             C=c,
#             solver='saga',
#             dual=False,
#             verbose=1,
#             random_state=69
#         )

#         model.fit(x_train_scaled, y_train)
        
#         # Calculate accuracies
#         train_pred = model.predict(x_train_scaled)
#         test_pred = model.predict(x_test_scaled)
        
#         train_acc = accuracy_score(y_train, train_pred)
#         test_acc = accuracy_score(y_test, test_pred)
        
#         train_accuracies.append(train_acc)
#         test_accuracies.append(test_acc)
        
#         print(f"\nRegularization Strength (C): {c} and Penalty: {p}")
#         print(f"Training accuracy: {train_acc:.4f}")
#         print(f"Test accuracy: {test_acc:.4f}")

# # Calculate errors (1 - accuracy)
# train_errors = [1 - acc for acc in train_accuracies]
# test_errors = [1 - acc for acc in test_accuracies]

# # Plot both accuracy and error on the same graph
# plt.figure(figsize=(10, 6))
# plt.plot(cs, train_accuracies, 'b-', label='Training Accuracy')
# plt.plot(cs, test_accuracies, 'r-', label='Test Accuracy')
# plt.plot(cs, train_errors, 'b--', label='Training Error')
# plt.plot(cs, test_errors, 'r--', label='Test Error')
# plt.xscale('log')
# plt.xlabel('Regularization Strength (C)')
# plt.ylabel('Rate')
# plt.title('Training and Test Accuracy/Error vs Regularization Strength')
# plt.legend()
# plt.grid(True)
# plt.savefig('logistic_regression_regularization_analysis.png', dpi=300, bbox_inches='tight')
# plt.show()






#ANALYSIS FOR DIFFERING TRAINING SIZE

data_sizes = [10000, 20000, 30000, 40000, 50000, 60000]
size_train_accuracies = []
size_test_accuracies = []
size_train_errors = []
size_test_errors = []

#Preset your parameters here
best_c = 1  
best_penalty = 'l2'  

for size in data_sizes:
    # Create a subset of the training data
    indices = np.random.choice(len(x_train_scaled), size, replace=False)
    x_train_subset = x_train_scaled[indices]
    y_train_subset = y_train[indices]
    
    # Train model on subset
    model = LogisticRegression(
        penalty=best_penalty,
        max_iter=500,
        C=best_c,
        solver='saga',
        dual=False,
        random_state=69
    )
    
    model.fit(x_train_subset, y_train_subset)
    
    # Calculate accuracies
    train_pred = model.predict(x_train_subset)
    test_pred = model.predict(x_test_scaled)
    
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
plt.xlabel('Training Data Size')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Performance vs Training Data Size')
plt.legend()
plt.grid(True)
plt.savefig('logistic_regression_data_size_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
    
