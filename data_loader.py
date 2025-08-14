import numpy as np
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

def load_data(subset=None):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Flatten and normalize
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    
    # Optional subset for faster training
    if subset:
        X_train, y_train = X_train[:subset], y_train[:subset]
        X_test, y_test = X_test[:subset], y_test[:subset]

    # One-hot encoding
    y_train_oh = np.zeros((y_train.size, 10))
    y_train_oh[np.arange(y_train.size), y_train] = 1
    y_test_oh = np.zeros((y_test.size, 10))
    y_test_oh[np.arange(y_test.size), y_test] = 1

    return X_train, y_train_oh, X_test, y_test_oh, y_train, y_test
