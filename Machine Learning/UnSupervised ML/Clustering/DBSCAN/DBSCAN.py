import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

# DATA LOAD
np.random.seed(42)

# Generate synthetic dataset with noise for DBSCAN
data = {
    'feature1': [],
    'feature2': []
}

# Create dense clusters with noise points
# Cluster 1
for i in range(150):
    f1 = np.random.normal(2, 0.5)
    f2 = np.random.normal(2, 0.5)
    data['feature1'].append(f1)
    data['feature2'].append(f2)

# Cluster 2
for i in range(150):
    f1 = np.random.normal(6, 0.6)
    f2 = np.random.normal(3, 0.6)
    data['feature1'].append(f1)
    data['feature2'].append(f2)

# Cluster 3
for i in range(150):
    f1 = np.random.normal(3, 0.4)
    f2 = np.random.normal(7, 0.5)
    data['feature1'].append(f1)
    data['feature2'].append(f2)

# Noise points
for i in range(100):
    f1 = np.random.uniform(0, 9)
    f2 = np.random.uniform(0, 9)
    data['feature1'].append(f1)
    data['feature2'].append(f2)

# Additional scattered points
for i in range(50):
    f1 = np.random.normal(8, 0.8)
    f2 = np.random.normal(7, 0.8)
    data['feature1'].append(f1)
    data['feature2'].append(f2)

df = pd.DataFrame(data)
copyData = df.copy()

print("=== DBSCAN Clustering ===")
print(f"Dataset shape: {copyData.shape}")
print(copyData.head())
print(f"\nDataset info:")
print(copyData.describe())

# MODEL
X = copyData[['feature1', 'feature2']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal epsilon using k-distance graph
k = 4  # MinPts = 4 (common choice)
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)
distances = np.sort(distances[:, k-1], axis=0)

# DBSCAN with different epsilon values
eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]
models = {}

print(f"\n=== Epsilon Parameter Tuning ===")
for eps in eps_values:
    model = DBSCAN(eps=eps, min_samples=k)
    clusters = model.fit_predict(X_scaled)
    
    # Calculate metrics only if we have more than one cluster (excluding noise)
    unique_labels = set(clusters)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(clusters).count(-1)
    
    models[eps] = {
        'model': model,
        'clusters': clusters,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }
    
    if n_clusters > 1:
        # Calculate silhouette score excluding noise points
        if n_noise < len(clusters):
            mask = clusters != -1
            silhouette_avg = silhouette_score(X_scaled[mask], clusters[mask]) if len(set(clusters[mask])) > 1 else 0
            davies_bouldin = davies_bouldin_score(X_scaled[mask], clusters[mask]) if len(set(clusters[mask])) > 1 else float('inf')
        else:
            silhouette_avg = 0
            davies_bouldin = float('inf')
    else:
        silhouette_avg = 0
        davies_bouldin = float('inf')
    
    models[eps]['silhouette'] = silhouette_avg
    models[eps]['davies_bouldin'] = davies_bouldin
    
    print(f"Eps: {eps}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    print(f"  Silhouette Score: {silhouette_avg:.4f}")

# Choose optimal epsilon (eps=0.5 typically works well)
optimal_eps = 0.5
best_model = models[optimal_eps]['model']
best_clusters = models[optimal_eps]['clusters']
copyData['cluster'] = best_clusters

# METRICS
n_clusters = models[optimal_eps]['n_clusters']
n_noise = models[optimal_eps]['n_noise']
silhouette_avg = models[optimal_eps]['silhouette']
davies_bouldin = models[optimal_eps]['davies_bouldin']

print(f"\n=== Best Model Metrics (eps={optimal_eps}) ===")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
if davies_bouldin != float('inf'):
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: K-distance graph for epsilon selection
plt.subplot(3, 3, 1)
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-NN Distance')
plt.title('K-Distance Graph (Elbow for Epsilon)')
plt.grid(True)

# Plot 2: Epsilon parameter comparison
plt.subplot(3, 3, 2)
eps_list = list(models.keys())
n_clusters_list = [models[eps]['n_clusters'] for eps in eps_list]
n_noise_list = [models[eps]['n_noise'] for eps in eps_list]

x = np.arange(len(eps_list))
width = 0.35

bars1 = plt.bar(x - width/2, n_clusters_list, width, label='Clusters', color='blue', alpha=0.7)
bars2 = plt.bar(x + width/2, n_noise_list, width, label='Noise Points', color='red', alpha=0.7)

plt.xlabel('Epsilon Values')
plt.ylabel('Count')
plt.title('Clusters vs Noise Points')
plt.xticks(x, eps_list)
plt.legend()

# Plot 3: Main clustering result
plt.subplot(3, 3, 3)
unique_labels = set(best_clusters)
colors = ['black'] + ['C{}'.format(i) for i in range(len(unique_labels)-1)]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'black'
        marker = 'x'
        label = 'Noise'
        alpha = 0.5
    else:
        marker = 'o'
        label = f'Cluster {k}'
        alpha = 0.8
    
    class_member_mask = (best_clusters == k)
    xy = X[class_member_mask]
    plt.scatter(xy.iloc[:, 0], xy.iloc[:, 1], c=col, marker=marker, alpha=alpha, s=60, label=label)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'DBSCAN Clustering (eps={optimal_eps}, min_samples={k})')
plt.legend()

# Plot 4: Core vs boundary vs noise points
plt.subplot(3, 3, 4)
core_samples_mask = np.zeros_like(best_clusters, dtype=bool)
core_samples_mask[best_model.core_sample_indices_] = True

colors = ['red', 'blue', 'green', 'purple', 'orange']
unique_labels = set(best_clusters)
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Noise points
        class_member_mask = (best_clusters == k)
        xy = X[class_member_mask]
        plt.scatter(xy.iloc[:, 0], xy.iloc[:, 1], c='black', marker='x', s=50, alpha=0.5, label='Noise')
    else:
        class_member_mask = (best_clusters == k)
        xy = X[class_member_mask]
        # Core points
        core_xy = xy[core_samples_mask[class_member_mask]]
        plt.scatter(core_xy.iloc[:, 0], core_xy.iloc[:, 1], c=col, marker='o', s=80, alpha=0.8, edgecolors='black')
        # Boundary points
        boundary_xy = xy[~core_samples_mask[class_member_mask]]
        plt.scatter(boundary_xy.iloc[:, 0], boundary_xy.iloc[:, 1], c=col, marker='o', s=30, alpha=0.5)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Core (large), Boundary (small), Noise (x)')

# Plot 5: Silhouette scores comparison
plt.subplot(3, 3, 5)
eps_list = list(models.keys())
silhouette_list = [models[eps]['silhouette'] for eps in eps_list]
plt.bar(eps_list, silhouette_list, color='green', alpha=0.7)
plt.xlabel('Epsilon Values')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Epsilon')

# Plot 6: Cluster distribution (excluding noise)
plt.subplot(3, 3, 6)
cluster_counts = pd.Series(best_clusters).value_counts().sort_index()
if -1 in cluster_counts.index:
    cluster_counts = cluster_counts.drop(-1)  # Remove noise count for this plot

if len(cluster_counts) > 0:
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    plt.bar(cluster_counts.index, cluster_counts.values, color=colors[:len(cluster_counts)])
    plt.xlabel('Cluster')
    plt.ylabel('Number of Points')
    plt.title('Cluster Distribution (Excluding Noise)')

# CUSTOM INPUT
print("\n=== Custom Point Classification ===")
print("Enter point coordinates for DBSCAN clustering:")
f1_input = float(input("Enter feature1 value (0-10): "))
f2_input = float(input("Enter feature2 value (0-10): "))

custom_data = np.array([[f1_input, f2_input]])
custom_data_scaled = scaler.transform(custom_data)

# For DBSCAN, we need to manually assign to nearest cluster or classify as noise
# Find distances to all core points
core_points = X_scaled[best_model.core_sample_indices_]
if len(core_points) > 0:
    distances_to_core = np.min(np.sqrt(np.sum((core_points - custom_data_scaled)**2, axis=1)))
    
    if distances_to_core <= optimal_eps:
        # Find which cluster this point would belong to
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(core_points)
        _, nearest_core_idx = nn.kneighbors(custom_data_scaled)
        actual_core_idx = best_model.core_sample_indices_[nearest_core_idx[0][0]]
        custom_cluster = best_clusters[actual_core_idx]
        point_type = "Core/Border"
    else:
        custom_cluster = -1
        point_type = "Noise"
else:
    custom_cluster = -1
    point_type = "Noise"

print(f"\nCustom Point Analysis:")
print(f"Feature 1: {f1_input:.2f}")
print(f"Feature 2: {f2_input:.2f}")
print(f"Assigned to Cluster: {custom_cluster}")
print(f"Point Type: {point_type}")

# Plot 7: Custom point visualization
plt.subplot(3, 3, 7)
unique_labels = set(best_clusters)
colors = ['black'] + ['C{}'.format(i) for i in range(len(unique_labels)-1)]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'black'
        marker = 'x'
        alpha = 0.5
    else:
        marker = 'o'
        alpha = 0.8
    
    class_member_mask = (best_clusters == k)
    xy = X[class_member_mask]
    plt.scatter(xy.iloc[:, 0], xy.iloc[:, 1], c=col, marker=marker, alpha=alpha, s=60)

plt.scatter(f1_input, f2_input, c='red', marker='X', s=300, label='Your Point', edgecolors='white', linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Your Point Classification')
plt.legend()

# Plot 8: Parameter sensitivity analysis
plt.subplot(3, 3, 8)
eps_range = np.arange(0.1, 1.0, 0.1)
cluster_counts = []
noise_counts = []

for eps in eps_range:
    temp_model = DBSCAN(eps=eps, min_samples=k)
    temp_clusters = temp_model.fit_predict(X_scaled)
    unique_labels = set(temp_clusters)
    n_clusters_temp = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise_temp = list(temp_clusters).count(-1)
    cluster_counts.append(n_clusters_temp)
    noise_counts.append(n_noise_temp)

plt.plot(eps_range, cluster_counts, 'b-o', label='Number of Clusters')
plt.plot(eps_range, noise_counts, 'r-s', label='Noise Points')
plt.xlabel('Epsilon')
plt.ylabel('Count')
plt.title('Parameter Sensitivity Analysis')
plt.legend()
plt.grid(True)

# Plot 9: Density visualization
plt.subplot(3, 3, 9)
# Create a heatmap of point density
from scipy.stats import gaussian_kde
xy = np.vstack([X['feature1'], X['feature2']])
kde = gaussian_kde(xy)
x_range = np.linspace(X['feature1'].min(), X['feature1'].max(), 50)
y_range = np.linspace(X['feature2'].min(), X['feature2'].max(), 50)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
Z = kde(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)

plt.contourf(X_grid, Y_grid, Z, levels=20, cmap='viridis', alpha=0.6)
plt.scatter(X['feature1'], X['feature2'], c=best_clusters, cmap='tab10', s=20, alpha=0.8)
if len(custom_data) > 0:
    plt.scatter(f1_input, f2_input, c='red', marker='X', s=300, label='Your Point', edgecolors='white', linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Density Distribution with Clusters')
plt.colorbar(label='Density')

plt.tight_layout()
plt.savefig('dbscan_clustering.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'dbscan_clustering.png'")