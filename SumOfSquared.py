import numpy as np

def k_means_one_iteration(data, initial_centroids):
    # Step 1: Assign clusters based on the nearest initial centroid
    clusters = np.array([np.argmin([np.abs(x - c) for c in initial_centroids]) for x in data])

    # Step 2: Calculate new centroids as the mean of assigned observations
    new_centroids = np.array([data[clusters == k].mean() for k in range(len(initial_centroids))])

    # Step 3: Calculate the total cost based on the new centroids
    cost = sum([(x - new_centroids[int(clusters[i])]) ** 2 for i, x in enumerate(data)])

    return clusters, new_centroids, cost

# Data and initial centroids
data = np.array([0.4, 0.5, 1.1, 2.2, 2.6, 3.0, 3.6, 3.7, 4.9, 5.0])
initial_centroids = np.array([0.66666667, 2.85,       4.53333333])

# Run one iteration of K-means
clusters, new_centroids, total_cost = k_means_one_iteration(data, initial_centroids)

print(f"Assigned clusters: {clusters}")
print(f"Updated centroids: {new_centroids}")
print(f"Total cost after update: {total_cost}")
