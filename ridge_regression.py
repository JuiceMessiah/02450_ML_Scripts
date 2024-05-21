import numpy as np

x = np.array([-3.4, -1.3, 0.5, 2.4, 4.2])
y = np.array([-2.9, -0.4, 0.7, 2.5, 4.5])
lambda_reg = 0.7
N = len(x)

mu = np.mean(x)
sigma_X = np.std(x, ddof=1)  # Sample standard deviation

x_standardized = (x - mu) / sigma_X

X = x_standardized.reshape(-1, 1)
X_transpose_X = X.T @ X
lambda_identity = lambda_reg * np.eye(X_transpose_X.shape[0])
inverse_term = np.linalg.inv(X_transpose_X + lambda_identity)
X_transpose_y = X.T @ y.reshape(-1, 1)

w1 = inverse_term @ X_transpose_y
w1 = w1.flatten()[0]

w0 = np.mean(y) - w1 * np.mean(x_standardized)

x2_standardized = x_standardized[1]  # Corresponding to -1.3
y_pred = w0 + w1 * x2_standardized

print(f"Mean of x (mu): {mu}")
print(f"Standard deviation of x (sigma_X): {sigma_X}")
print(f"Standardized x: {x_standardized}")
print(f"Calculated w0: {w0}")
print(f"Calculated w1: {w1}")
print(f"Prediction for x2 (standardized): {y_pred}")