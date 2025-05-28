import torch
from torchvision import datasets, transforms
from collections import Counter
import matplotlib.pyplot as plt
# This script explores the label distribution in the FashionMNIST dataset.

# Load the FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Extract labels
labels = train_dataset.targets.numpy()

# Count occurrences of each label
label_counts = Counter(labels)

# Total number of labels
total_labels = len(labels)

# Calculate percentage for each label
label_percentages = {label: (count / total_labels) * 100 for label, count in label_counts.items()}

# Plot the label distribution
plt.bar(label_percentages.keys(), label_percentages.values())
plt.xlabel('Labels')
plt.ylabel('Percentage (%)')
plt.title('Label Distribution in FashionMNIST')
plt.xticks(range(10), [str(i) for i in range(10)])  # Set x-ticks to label numbers
plt.grid(axis='y')
plt.savefig('label_distribution.png')
plt.show()

# Print results
print("Label Distribution:")
for label, percentage in label_percentages.items():
    print(f"Label {label}: {percentage:.2f}%")