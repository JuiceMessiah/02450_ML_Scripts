import numpy as np
from sklearn.linear_model import Ridge

# Given data
x = np.array([-0.5, 0.39, 1.19, -1.08])
y = np.array([-0.86, -0.61, 1.37, 0.1])

# Weights for ridge regression
alpha = 0.25

# Transform xi to [xi, xi^2]
X_transformed = np.vstack((x, x**3)).T

# Transform xi to [xi, xi^2]
# X_transformed = np.vstack((x, np.sin(x))).T

# Compute mean and standard deviation for standardization
mu = X_transformed.mean(axis=0)
sigma = X_transformed.std(axis=0, ddof=1)
X_std = (X_transformed - mu) / sigma

# Fit ridge regression model
ridge_model = Ridge(alpha=alpha, fit_intercept=True)
ridge_model.fit(X_std, y)
w0 = ridge_model.intercept_
w = ridge_model.coef_

regulization = alpha * np.sum(w**2)


# Prediction function
def predict(X, w0, w):
    return w0 + X @ w


# Compute predictions
y_pred = predict(X_std, w0, w)

# Compute sum of squared errors
sse = np.sum((y - y_pred)**2)

# Output the results
print(f"Regularization term: {regulization:.4f}")
print("Mean of transformed X (mu):", mu)
print("Standard deviation of transformed X (sigma):", sigma)
print("Standardized transformed X (X_std):", X_std)
print("Intercept (w0):", w0)
print("Weights (w):", w)
print("Predicted values (y_pred):", y_pred)
print("Sum of squared errors (SSE):", sse)
print(f"Total loss: {sse + regulization:.4f}")
print("Which of the total loss values is closest to your E__lambda value? \n")


def create_contingency_matrix(distance_matrix, cutoff):
    """
    Create a contingency matrix from a distance matrix by applying a cutoff.

    Parameters:
        distance_matrix (np.ndarray): The symmetric matrix of distances.
        cutoff (float): The threshold to decide whether two elements are in the same cluster.

    Returns:
        np.ndarray: A binary matrix where entry (i, j) is 1 if elements i and j are in the same cluster, else 0.
    """
    return (distance_matrix <= cutoff).astype(int)

def rand_index(contingency_matrix):
    """
    Calculate the Rand Index for a clustering given by a contingency matrix.

    Parameters:
        contingency_matrix (np.ndarray): A binary matrix where entry (i, j) indicates
                                         whether elements i and j are in the same cluster.

    Returns:
        float: The Rand Index score.
    """
    # Total pairs of elements
    n = contingency_matrix.shape[0]
    total_pairs = n * (n - 1) / 2

    # True Positive (TP): both in the same cluster in predicted and true labels
    # True Negative (TN): both in different clusters in predicted and true labels
    # Sum all elements where both matrices have 1s (intersection)
    TP_plus_FP = np.sum(contingency_matrix.sum(axis=1) * (contingency_matrix.sum(axis=1) - 1) / 2)
    TP = np.sum(contingency_matrix * contingency_matrix.T) / 2

    # Adjustment for false positives
    FP = TP_plus_FP - TP

    # Compute TN and FN using complement properties
    TN_plus_FN = total_pairs - TP_plus_FP
    TN = total_pairs + TP - np.sum(contingency_matrix.sum(axis=0) * contingency_matrix.sum(axis=1))
    FN = TN_plus_FN - TN

    # Rand Index
    return (TP + TN) / (TP + TN + FP + FN)

# Example usage
distance_matrix = np.array([
    [0, 0.3, 0.4, 0.7],
    [0.3, 0, 0.5, 0.8],
    [0.4, 0.5, 0, 0.6],
    [0.7, 0.8, 0.6, 0]
])

cutoff = 0.5  # Define your cutoff here
contingency_matrix = create_contingency_matrix(distance_matrix, cutoff)
rand_score = rand_index(contingency_matrix)

print("Contingency Matrix:\n", contingency_matrix)
print("Rand Index:", rand_score)

