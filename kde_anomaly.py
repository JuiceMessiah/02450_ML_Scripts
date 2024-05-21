import numpy as np
from sklearn.neighbors import KernelDensity

# Training data
X_train = np.array([-3, -1, 5, 6]).reshape(-1, 1)

# Test points, ÆNDRER DENNE VÆRDI
X_test = np.array([-4, 2]).reshape(-1, 1)

# Kernel Density Estimator with bandwidth = sqrt(0.5)
kde = KernelDensity(kernel='gaussian', bandwidth=np.sqrt(0.5)).fit(X_train)

# Estimate the density at the test points
log_density = kde.score_samples(X_test)
density = np.exp(log_density)

# Check if densities are below the anomaly threshold of 0.015
anomalies = density < 0.015

# Print the densities and anomaly status
print(f"Densities at test points: {density}")
print(f"Anomaly status (True means anomaly): {anomalies}")

# Determine which statement is correct
if all(anomalies):
    print("A. Both test observations are anomalies.")
elif not anomalies[0] and anomalies[1]:
    print("B. The observation at x = -4 is not an anomaly but the observation at x = 2 is an anomaly.")
elif anomalies[0] and not anomalies[1]:
    print("C. The observation at x = -4 is an anomaly but the observation at x = 2 is not an anomaly.")
elif not any(anomalies):
    print("D. Neither of the test observations are anomalies.")
else:
    print("E. Don’t know.")
