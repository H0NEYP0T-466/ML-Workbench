import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import seaborn as sns

# DATA LOAD
np.random.seed(42)

# Generate synthetic dataset for hierarchical clustering
data = {
    'feature1': [],
    'feature2': [],
    'feature3': []
}

# Create distinct groups for hierarchical structure
for i in range(600):
    if i < 150:  # Group 1
        f1 = np.random.normal(2, 0.8)
        f2 = np.random.normal(2, 0.8)
        f3 = np.random.normal(1, 0.5)
    elif i < 300:  # Group 2
        f1 = np.random.normal(6, 1.0)
        f2 = np.random.normal(3, 0.9)
        f3 = np.random.normal(5, 0.7)
    elif i < 450:  # Group 3
        f1 = np.random.normal(3, 0.7)
        f2 = np.random.normal(7, 1.1)
        f3 = np.random.normal(3, 0.6)
    else:  # Group 4
        f1 = np.random.normal(8, 0.9)
        f2 = np.random.normal(7, 0.8)
        f3 = np.random.normal(6, 0.8)
    
    data['feature1'].append(f1)
    data['feature2'].append(f2)
    data['feature3'].append(f3)

df = pd.DataFrame(data)
copyData = df.copy()

print("=== Hierarchical Clustering (Agglomerative) ===")
print(f"Dataset shape: {copyData.shape}")
print(copyData.head())
print(f"\nDataset info:")
print(copyData.describe())

# MODEL
X = copyData[['feature1', 'feature2', 'feature3']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical clustering with different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
n_clusters = 4

print(f"\n=== Linkage Method Comparison ===")
models = {}
for method in linkage_methods:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    clusters = model.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    
    models[method] = {
        'model': model,
        'clusters': clusters,
        'silhouette': silhouette_avg,
        'davies_bouldin': davies_bouldin
    }
    
    print(f"{method.capitalize()} linkage:")
    print(f"  Silhouette Score: {silhouette_avg:.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")

# Choose best method (typically ward for well-separated clusters)
best_method = 'ward'
best_model = models[best_method]['model']
best_clusters = models[best_method]['clusters']
copyData['cluster'] = best_clusters

# METRICS
silhouette_avg = models[best_method]['silhouette']
davies_bouldin = models[best_method]['davies_bouldin']

print(f"\n=== Best Model Metrics ({best_method} linkage) ===")
print(f"Number of clusters: {n_clusters}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: Dendrogram
plt.subplot(3, 3, 1)
# Create linkage matrix for dendrogram
linkage_matrix = linkage(X_scaled, method=best_method)
dendrogram(linkage_matrix, truncate_mode='level', p=10)
plt.title(f'Dendrogram ({best_method.capitalize()} Linkage)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# Plot 2: Linkage method comparison
plt.subplot(3, 3, 2)
methods = list(models.keys())
silhouette_scores = [models[m]['silhouette'] for m in methods]
colors = ['blue', 'orange', 'green', 'red']
bars = plt.bar(methods, silhouette_scores, color=colors)
plt.xlabel('Linkage Method')
plt.ylabel('Silhouette Score')
plt.title('Linkage Method Comparison')
plt.xticks(rotation=45)
for bar, score in zip(bars, silhouette_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom')

# Plot 3: Feature1 vs Feature2
plt.subplot(3, 3, 3)
scatter = plt.scatter(copyData['feature1'], copyData['feature2'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Feature 1 vs Feature 2')
plt.colorbar(scatter)

# Plot 4: Feature1 vs Feature3
plt.subplot(3, 3, 4)
scatter = plt.scatter(copyData['feature1'], copyData['feature3'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 3')
plt.title('Feature 1 vs Feature 3')
plt.colorbar(scatter)

# Plot 5: Feature2 vs Feature3
plt.subplot(3, 3, 5)
scatter = plt.scatter(copyData['feature2'], copyData['feature3'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Feature 2')
plt.ylabel('Feature 3')
plt.title('Feature 2 vs Feature 3')
plt.colorbar(scatter)

# Plot 6: Cluster distribution
plt.subplot(3, 3, 6)
cluster_counts = copyData['cluster'].value_counts().sort_index()
colors = ['blue', 'orange', 'green', 'red']
plt.bar(cluster_counts.index, cluster_counts.values, color=colors[:len(cluster_counts)])
plt.xlabel('Cluster')
plt.ylabel('Number of Points')
plt.title('Cluster Distribution')

# CUSTOM INPUT
print("\n=== Custom Point Classification ===")
print("Enter point coordinates for hierarchical clustering:")
f1_input = float(input("Enter feature1 value (0-10): "))
f2_input = float(input("Enter feature2 value (0-10): "))
f3_input = float(input("Enter feature3 value (0-10): "))

custom_data = np.array([[f1_input, f2_input, f3_input]])
custom_data_scaled = scaler.transform(custom_data)

# For hierarchical clustering, we need to find the closest cluster
# Since AgglomerativeClustering doesn't have predict method, we'll use distance to cluster centers
cluster_centers = []
for i in range(n_clusters):
    cluster_points = X_scaled[best_clusters == i]
    center = np.mean(cluster_points, axis=0)
    cluster_centers.append(center)

cluster_centers = np.array(cluster_centers)
distances = np.sqrt(np.sum((cluster_centers - custom_data_scaled)**2, axis=1))
custom_cluster = np.argmin(distances)

print(f"\nCustom Point Analysis:")
print(f"Feature 1: {f1_input:.2f}")
print(f"Feature 2: {f2_input:.2f}")
print(f"Feature 3: {f3_input:.2f}")
print(f"Assigned to Cluster: {custom_cluster}")
print(f"Distance to cluster center: {distances[custom_cluster]:.4f}")

# Plot 7: Custom point visualization
plt.subplot(3, 3, 7)
scatter = plt.scatter(copyData['feature1'], copyData['feature2'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.scatter(f1_input, f2_input, c='black', marker='X', s=300, label='Your Point', edgecolors='white', linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Your Point Classification')
plt.legend()
plt.colorbar(scatter)

# Plot 8: Cluster characteristics heatmap
plt.subplot(3, 3, 8)
cluster_means = copyData.groupby('cluster')[['feature1', 'feature2', 'feature3']].mean()
sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'Average Value'})
plt.title('Cluster Characteristics')
plt.ylabel('Features')

# Plot 9: 3D visualization using PCA
plt.subplot(3, 3, 9)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=copyData['cluster'], cmap='viridis', alpha=0.7)
if len(custom_data) > 0:
    custom_pca = pca.transform(custom_data_scaled)
    plt.scatter(custom_pca[:, 0], custom_pca[:, 1], c='black', marker='X', s=300, label='Your Point', edgecolors='white', linewidths=2)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA Projection of Clusters')
plt.legend()
plt.colorbar(scatter)

plt.tight_layout()
plt.savefig('hierarchical_clustering.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'hierarchical_clustering.png'")