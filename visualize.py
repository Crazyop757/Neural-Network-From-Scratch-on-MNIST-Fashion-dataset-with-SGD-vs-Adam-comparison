# visualize.py
import matplotlib.pyplot as plt
import numpy as np

# Fashion-MNIST class labels
LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def plot_loss_accuracy(history_dict):
    epochs = range(1, len(next(iter(history_dict.values()))["loss"]) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for optimizer, history in history_dict.items():
        plt.plot(epochs, history["loss"], label=f"{optimizer}")
    plt.title("Training Loss Comparison (SGD vs Adam)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for optimizer, history in history_dict.items():
        plt.plot(epochs, history["accuracy"], label=f"{optimizer}")
    plt.title("Training Accuracy Comparison (SGD vs Adam)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()



def plot_predictions(X, y_true, y_pred, num_samples=10):
    """Visualize predictions with true labels."""
    indices = np.random.choice(len(X), num_samples, replace=False)
    plt.figure(figsize=(15, 3))

    for i, idx in enumerate(indices):
        image = X[idx].reshape(28, 28)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title(f"True: {LABELS[y_true[idx]]}\nPred: {LABELS[y_pred[idx]]}")

    plt.show()


def visualize_weights(weights):
    """Visualize learned weights of the first hidden layer as images."""
    W = weights[0]  # first layer weights (shape: hidden_size x input_size)
    num_neurons = min(10, W.shape[0])  # visualize up to 10 neurons

    plt.figure(figsize=(15, 4))
    for i in range(num_neurons):
        plt.subplot(1, num_neurons, i + 1)
        plt.imshow(W[i].reshape(28, 28), cmap="seismic", interpolation="nearest")
        plt.axis("off")
        plt.title(f"Neuron {i+1}")
    plt.show()
