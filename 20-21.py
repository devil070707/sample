import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_centroids(data, k):
    """Initialize k centroids by randomly selecting k data points"""
    n_samples = len(data)
    random_indices = random.sample(range(n_samples), k)
    return data[random_indices]

def assign_clusters(data, centroids):
    """Assign each point to the nearest centroid"""
    n_samples = len(data)
    cluster_labels = np.zeros(n_samples)
    
    for i in range(n_samples):
        distances = [euclidean_distance(data[i], centroid) for centroid in centroids]
        cluster_labels[i] = np.argmin(distances)
    
    return cluster_labels

def update_centroids(data, cluster_labels, k):
    """Update centroids based on mean of points in each cluster"""
    n_features = data.shape[1]
    centroids = np.zeros((k, n_features))
    
    for i in range(k):
        cluster_points = data[cluster_labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
    
    return centroids

def kmeans(data, k, max_iterations=10):
    """Perform K-means clustering"""
    # Initialize centroids
    centroids = initialize_centroids(data, k)
    
    for iteration in range(max_iterations):
        # Assign points to clusters
        cluster_labels = assign_clusters(data, centroids)
        
        # Update centroids
        new_centroids = update_centroids(data, cluster_labels, k)
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
        
    return centroids, cluster_labels

# Load and prepare the Iris dataset
data = pd.read_csv('IRIS.csv')  # Adjust path as needed
X = data.iloc[:, :-1].values  # Select all features except species

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run K-means with K=3
print("\nK-means Clustering with K=3:")
centroids_k3, labels_k3 = kmeans(X_scaled, k=3)
print("\nFinal Cluster Means (K=3):")
# Convert scaled centroids back to original scale
centroids_original_k3 = scaler.inverse_transform(centroids_k3)
for i, centroid in enumerate(centroids_original_k3):
    print(f"Cluster {i + 1}: {centroid}")

# Run K-means with K=4
print("\nK-means Clustering with K=4:")
centroids_k4, labels_k4 = kmeans(X_scaled, k=4)
print("\nFinal Cluster Means (K=4):")
# Convert scaled centroids back to original scale
centroids_original_k4 = scaler.inverse_transform(centroids_k4)
for i, centroid in enumerate(centroids_original_k4):
    print(f"Cluster {i + 1}: {centroid}")