import matplotlib.pyplot as plt
import numpy as np

# Example: Assume you have fewer epochs and metrics to store (e.g., 10 epochs)
epochs = np.arange(1, 11)  # 10 epochs
train_loss = np.random.random(10)  # Random example loss values, replace with actual data
train_accuracy = np.random.random(10) * 100  # Random accuracy, replace with actual values

# Plotting training loss and accuracy
plt.figure(figsize=(14, 6))

# Subplot for Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="Training Loss", color='r', marker='o', markersize=6)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Subplot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, label="Training Accuracy", color='g', marker='o', markersize=6)
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()
