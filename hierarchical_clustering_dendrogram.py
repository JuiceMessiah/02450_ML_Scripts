import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

def create_symmetric_matrix(triangular_values):
    # Determine the size of the full matrix
    n = len(triangular_values) + 1  # Add one because the input starts from the first off-diagonal elements

    # Create an empty matrix of the appropriate size
    matrix = np.zeros((n, n))

    # Fill the matrix symmetrically from the triangular values
    for i in range(1, n):
        for j in range(i):
            # Populate both the lower and the upper parts of the matrix
            matrix[i, j] = triangular_values[i - 1][j]
            matrix[j, i] = triangular_values[i - 1][j]

    # Set diagonal values to 0, which are zeros by default but can be set explicitly if needed
    for k in range(n):
        matrix[k, k] = 0.0

    return matrix

# Example input
triangular_input = [
    [2.6],
    [2.8, 1.8],
    [2.1, 2.3, 1.2],
    [0.8, 2.7, 2.9, 2.4],
    [1.7, 1.4, 2.1, 1.8, 2.0],
    [3.9, 1.9, 2.2, 2.6, 4.2, 2.4],
    [3.8, 1.7, 1.6, 2.5, 4.0, 2.6, 1.2],
    [4.1, 2.0, 2.3, 3.0, 4.4, 2.7, 0.6, 1.0],
    [4.2, 2.2, 2.0, 2.9, 4.5, 3.0, 1.3, 0.6, 0.9]
]
# Create the matrix
distance_matrix = create_symmetric_matrix(triangular_input)

# Print the full matrix with formatting
print("Symmetric Matrix:")

for row in distance_matrix:
    print(" ".join(f"{num:.1f}" for num in row))

# Convert the symmetric distance matrix to a condensed distance matrix
condensed_matrix = squareform(distance_matrix)

# different linkage methods ['single', 'complete', 'average', 'centroid', 'ward']
# Perform hierarchical clustering using complete linkage
linked = linkage(condensed_matrix, method='complete')

# Define the observation labels
labels = [f"o{i+1}" for i in range(len(distance_matrix))]

# Plot the dendrogram
plt.figure(figsize=(14, 10))
dendro = dendrogram(linked,
                    orientation='top',
                    distance_sort='ascending',
                    show_leaf_counts=True,
                    labels=labels,
                    above_threshold_color='black',
                    color_threshold=0.0)

# Add the distance values to the dendrogram
for i, d, c in zip(dendro['icoord'], dendro['dcoord'], dendro['color_list']):
    x = 0.5 * sum(i[1:3])
    y = d[1]
    plt.plot(x, y, 'o', c=c)
    plt.annotate(f'{y:.2f}', (x, y), xytext=(0, -8),
                 textcoords='offset points',
                 va='top', ha='center', fontsize=8, color=c)

plt.title("Dendrogram (Complete Linkage)")
plt.xlabel("Observations")
plt.ylabel("Euclidean Distance")
plt.grid(True)
plt.show()
