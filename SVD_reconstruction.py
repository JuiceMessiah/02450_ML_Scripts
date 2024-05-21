import numpy as np

X = np.array([[-0.6, -0.6, 2.5, -0.1],
             [-0.8, -0.3, -1, 1.2],
             [-0.7, 0.3, -0.2, -0.1],
             [1.4, 1, 0.1, -2.8],
             [-0.2, 0.8, -1.2, 0.7]])

# Calculate mean
mean = np.mean(X, axis=0)
print(mean)

V = np.array([[0.43, -0.26],
            [0.17, -0.37],
            [0.33, 0.88],
            [-0.82, 0.14]])

x = np.array([1.4, 1, 0.1, -2.8])

hat_x = x - mean
print(hat_x)

projection = np.dot(hat_x, V)
print(projection)

hat_x_4_reconstructed = np.dot(projection, V.T) + mean
print(hat_x_4_reconstructed)