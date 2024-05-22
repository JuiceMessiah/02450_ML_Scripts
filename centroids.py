import numpy as np

# Updated custom k-means function to track centroid locations
def custom_kmeans(data, initial_centers, num_iterations):
    # Convert data to 2D array if it's a 1D array
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    # Convert initial centers to 2D array if it's a 1D array
    if initial_centers.ndim == 1:
        initial_centers = initial_centers.reshape(-1, 1)
    
    # Initialize centroids with the provided initial centers
    centroids = initial_centers
    centroid_history = [centroids.flatten().tolist()]  # Track the history of centroids
    
    for i in range(num_iterations):
        # Assign each sample to the nearest centroid
        labels = np.argmin(np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)
        
        # Calculate new centroids as the mean of the points in each cluster
        new_centroids = np.array([data[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] 
                                  for j in range(len(centroids))])
        centroid_history.append(new_centroids.flatten().tolist())
        
        # If centroids do not change, we have converged and can break early
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    # Return the final clusters and centroid history
    clusters = {i: [] for i in range(len(centroids))}
    for label, point in zip(labels, data):
        clusters[label].append(point[0])
    
    # Sort clusters for display
    sorted_clusters = [sorted(cluster) for cluster in clusters.values()]
    
    return sorted_clusters, centroid_history

# Dataset and initial cluster centers
data = np.array([0, 2, 4, 5, 6, 7, 14])
initial_centers = np.array([0, 5, 8])

# Number of iterations for centroid computation
t1 = 1
t2 = 2

# Perform custom k-means clustering for specified iterations
clusters_t1, centroids_t1 = custom_kmeans(data, initial_centers, t1)
clusters_t2, centroids_t2 = custom_kmeans(data, initial_centers, t2)

# Print the resulting clusters and centroid locations
print("Clusters after 1 iteration:", clusters_t1)
print("Centroid locations after each update for 1 iteration:", centroids_t1)
print("Clusters after 2 iterations:", clusters_t2)
print("Centroid locations after each update for 2 iterations:", centroids_t2)


