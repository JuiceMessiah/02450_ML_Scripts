import numpy as np

n = 5
x = np.array([-3.4, -1.3, 0.5, 2.4, 4.2])
y = np.array([-2.9, -0.4, 0.7, 2.5, 4.5])
lam = 0.7

std = np.sqrt((1/4) * np.sum((x - np.mean(x))**2))
print(x.mean())
print(std)
x_standardized = (x - np.mean(x)) / std

print(x_standardized)

x_tilde = np.vstack((np.ones(x_standardized.shape[0]), x_standardized)).T

reg_matrix = np.diag([0, 1]) * lam

w_star = np.linalg.inv(x_tilde.T @ x_tilde + reg_matrix) @ x_tilde.T @ y

print(w_star)

y_pred = w_star[1] * x_standardized + w_star[0]

print(y_pred)