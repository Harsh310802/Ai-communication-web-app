import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate 26 labels from 'a' to 'z'
labels = [chr(i) for i in range(97, 123)]  # 'a' to 'z'

# Generate random true and predicted labels for 100 samples
true_labels = np.random.choice(labels, size=100)
predicted_labels = np.random.choice(labels, size=100)

# Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

# Create a heatmap of the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=True)

# Rotate labels for better visibility
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# Label the axes and the plot
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Labels a to z')
plt.show()
