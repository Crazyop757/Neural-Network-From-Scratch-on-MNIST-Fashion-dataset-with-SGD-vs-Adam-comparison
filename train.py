import numpy as np
from model import NeuralNet
from loss import cross_entropy
from optimizers import SGD, Adam
from data_loader import load_data
from visualize import plot_loss_accuracy, plot_predictions, visualize_weights


def accuracy(y_pred, y_true_labels):
    return np.mean(np.argmax(y_pred, axis=1) == y_true_labels)


def train_model(model, optimizer, X_train, y_train_oh, X_test, y_test_labels, epochs=10):
    losses, accs = [], []
    for epoch in range(epochs):
        # Forward + loss
        out = model.forward(X_train)
        loss = cross_entropy(out, y_train_oh)

        # Backward + update
        grads = model.backward(X_train, y_train_oh)
        optimizer.step(model, grads)

        preds = model.forward(X_test)
        acc = accuracy(preds, y_test_labels)

        losses.append(loss)
        accs.append(acc)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Acc: {acc:.4f}")
    return losses, accs


if __name__ == "__main__":
    X_train, y_train_oh, X_test, y_test_oh, y_train_labels, y_test_labels = load_data(subset=5000)

    # SGD Training
    print("Training with SGD...")
    model_sgd = NeuralNet()
    sgd_losses, sgd_accs = train_model(model_sgd, SGD(lr=0.01), 
                                       X_train, y_train_oh, X_test, y_test_labels, epochs=20)

    # Adam Training
    print("Training with Adam...")
    model_adam = NeuralNet()
    adam_losses, adam_accs = train_model(model_adam, Adam(lr=0.001), 
                                         X_train, y_train_oh, X_test, y_test_labels, epochs=20)

    np.save("sgd_accs.npy", sgd_accs)
    np.save("adam_accs.npy", adam_accs)

sgd_history = {"loss": sgd_losses, "accuracy": sgd_accs}
adam_history = {"loss": adam_losses, "accuracy": adam_accs}

history_dict = {
        "SGD": sgd_history,
        "Adam": adam_history
                }

# --- Use visualize.py functions ---
print("\nPlotting training curves for Adam optimizer...")
plot_loss_accuracy(history_dict)

print("\nEvaluating predictions on test set...")
preds = np.argmax(model_adam.forward(X_test), axis=1)
plot_predictions(X_test, y_test_labels, preds)

print("\nVisualizing first layer weights...")
visualize_weights(model_adam.weights)
