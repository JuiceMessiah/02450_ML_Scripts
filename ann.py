import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)


def neural_network(x, w1, w2, w2_0):
    # Compute the first layer output
    h1_input = np.dot(w1[:, 1:], x) + w1[:, 0]
    h1_output = tanh(h1_input)

    # Compute the second layer output
    h2_input = np.dot(w2[1:], h1_output) + w2[0]
    output = tanh(h2_input + w2_0)  # linear activation function h^(2)(x) =s

    return output


# Weights from the picture
w1 = np.array([
    [2.2, 0.7, -0.3],
    [-0.2, 0.8, 0.4]
])

w2 = np.array([2.2, -0.7, 0.5])

# User input for x values
x1 = -2
x2 = -2

x = np.array([x1, x2])

# Calculate the network output
output = neural_network(x, w1, w2, 0)

print("The output of the neural network is:", output)