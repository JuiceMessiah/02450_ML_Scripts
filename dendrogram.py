import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch

# Your hardcoded distance matrix
distance_matrix = np.array([
    [0.0, 2.6, 2.8, 2.1, 0.8, 1.7, 3.9, 3.8, 4.1, 4.2],
    [2.6, 0.0, 1.8, 2.3, 2.7, 1.4, 3.9, 3.8, 4.1, 4.2],
    [2.8, 1.8, 0.0, 1.2, 2.9, 2.1, 1.9, 1.6, 2.3, 2.0],
    [2.1, 2.3, 1.2, 0.0, 2.4, 1.8, 2.6, 2.5, 3.0, 2.9],
    [0.8, 2.7, 2.9, 2.4, 0.0, 2.0, 4.2, 4.0, 4.4, 4.5],
    [1.7, 1.4, 2.1, 1.8, 2.0, 0.0, 2.4, 2.6, 2.7, 3.0],
    [3.9, 3.9, 1.9, 2.6, 4.2, 2.4, 0.0, 1.2, 0.6, 1.3],
    [3.8, 3.8, 1.6, 2.5, 4.0, 2.6, 1.2, 0.0, 1.0, 0.9],
    [4.1, 4.1, 2.3, 3.0, 4.4, 2.7, 0.6, 1.0, 0.0, 0.9],
    [4.2, 4.2, 2.0, 2.9, 4.5, 3.0, 1.3, 0.9, 0.9, 0.0]
])

# Ensure the matrix is symmetric and has zero diagonals
assert np.allclose(distance_matrix, distance_matrix.T), "Matrix is not symmetric"
assert np.allclose(np.diag(distance_matrix), np.zeros(distance_matrix.shape[0])), "Diagonal should be zero"

# Convert to condensed matrix
condensed_matrix = squareform(distance_matrix)

# Choose 'single' for minimum linkage and 'complete' for maximum linkage
linkage_type = 'complete'  # Or 'complete'

# Generate the dendrogram using maximum linkage
linked_corrected = sch.linkage(condensed_matrix, 'complete')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked_corrected,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           labels=np.arange(1, 11))  # Set labels to count from 1 to 10

# Set y-axis ticks with 0.1 intervals
max_distance = np.max(linked_corrected[:, 2])
plt.yticks(np.arange(0, max_distance + 0.1, 0.1))

plt.title(f'Dendrogram using {linkage_type} linkage')
plt.xlabel('Index of observations')
plt.ylabel('Distance')
plt.grid()
plt.show()
