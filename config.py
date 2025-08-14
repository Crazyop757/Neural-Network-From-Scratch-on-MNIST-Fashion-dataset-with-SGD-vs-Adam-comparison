# config.py

# Training hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64

# Network architecture
INPUT_SIZE = 784      # 28x28 images flattened
HIDDEN_SIZE = 128     # number of neurons in hidden layer
OUTPUT_SIZE = 10      # 10 classes in Fashion-MNIST

# Regularization
DROPOUT_RATE = 0.2
LAMBDA = 0.001  # L2 regularization strength

# Random seed for reproducibility
SEED = 42
