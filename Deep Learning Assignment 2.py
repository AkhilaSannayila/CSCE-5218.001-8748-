import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of activation functions
def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases
np.random.seed(42)
w1 = np.random.randn(2, 2)  # weights for hidden layer
b1 = np.random.randn(2)    # biases for hidden layer
w2 = np.random.randn(2, 1)  # weights for output layer
b2 = np.random.randn(1)    # bias for output layer

# Training data (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Hyperparameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X, w1) + b1
    h1 = relu(z1)
    z2 = np.dot(h1, w2) + b2
    y_pred = sigmoid(z2)

    # Compute loss
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # Backpropagation
    d_loss = (y_pred - y) / y_pred.shape[0]
    d_z2 = d_loss * sigmoid_derivative(y_pred)
    d_w2 = np.dot(h1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0)

    d_h1 = np.dot(d_z2, w2.T)
    d_z1 = d_h1 * relu_derivative(h1)
    d_w1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0)

    # Update weights and biases
    w2 -= learning_rate * d_w2
    b2 -= learning_rate * d_b2
    w1 -= learning_rate * d_w1
    b1 -= learning_rate * d_b1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Test the network
z1 = np.dot(X, w1) + b1
h1 = relu(z1)
z2 = np.dot(h1, w2) + b2
y_pred = sigmoid(z2)

print("Predictions:")
print(y_pred)