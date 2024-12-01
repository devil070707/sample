import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

def assign_clusters(points, centroids):
    clusters = []
    for point in points:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return clusters

def update_centroids(points, clusters, k):
    centroids = []
    for i in range(k):
        cluster_points = [p for j, p in enumerate(points) if clusters[j] == i]
        if cluster_points:
            centroid = np.mean(cluster_points, axis=0)
        else:
            centroid = points[i]  # Keep old centroid if cluster is empty
        centroids.append(centroid)
    return centroids

# Problem 9
print("Problem 9:")
points1 = np.array([
    [0.1, 0.6],   # P1
    [0.15, 0.71], # P2
    [0.08, 0.9],  # P3
    [0.16, 0.85], # P4
    [0.2, 0.3],   # P5
    [0.25, 0.5],  # P6
    [0.24, 0.1],  # P7
    [0.3, 0.2]    # P8
])

# Initial centroids
m1 = points1[0]  # P1
m2 = points1[7]  # P8
centroids1 = [m1, m2]

# Assign clusters
clusters1 = assign_clusters(points1, centroids1)

# Get updated centroids
new_centroids1 = update_centroids(points1, clusters1, 2)

print("\n1. P6 belongs to Cluster #:", clusters1[5] + 1)
print("2. Population around m2 (Cluster 2):", clusters1.count(1))
print("3. Updated centroids:")
print(f"m1: [{new_centroids1[0][0]:.4f}, {new_centroids1[0][1]:.4f}]")
print(f"m2: [{new_centroids1[1][0]:.4f}, {new_centroids1[1][1]:.4f}]")

# Problem 10
print("\nProblem 10:")
points2 = np.array([
    [2, 10], # P1
    [2, 5],  # P2
    [8, 4],  # P3
    [5, 8],  # P4
    [7, 5],  # P5
    [6, 4],  # P6
    [1, 2],  # P7
    [4, 9]   # P8
])

# Initial centroids
m1 = points2[0]  # P1
m2 = points2[3]  # P4
m3 = points2[6]  # P7
centroids2 = [m1, m2, m3]

# Assign clusters
clusters2 = assign_clusters(points2, centroids2)

# Get updated centroids
new_centroids2 = update_centroids(points2, clusters2, 3)

print("\n1. P6 belongs to Cluster #:", clusters2[5] + 1)
print("2. Population around m3 (Cluster 3):", clusters2.count(2))
print("3. Updated centroids:")
print(f"m1: [{new_centroids2[0][0]:.4f}, {new_centroids2[0][1]:.4f}]")
print(f"m2: [{new_centroids2[1][0]:.4f}, {new_centroids2[1][1]:.4f}]")
print(f"m3: [{new_centroids2[2][0]:.4f}, {new_centroids2[2][1]:.4f}]")