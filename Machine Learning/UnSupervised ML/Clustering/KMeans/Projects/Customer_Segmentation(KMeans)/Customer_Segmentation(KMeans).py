import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# DATA LOAD
data = pd.read_csv('customer_segmentation_dataset.csv')
copyData = data.copy()

print("=== Customer Segmentation with KMeans ===")
print(f"Dataset shape: {copyData.shape}")
print(copyData.head())
print(f"\nDataset info:")
print(copyData.describe())

# MODEL
X = copyData[['age', 'annual_income', 'spending_score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Choose optimal k (k=4 based on analysis)
optimal_k = 4
model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)
copyData['cluster'] = clusters

# METRICS
silhouette_avg = silhouette_score(X_scaled, clusters)
davies_bouldin = davies_bouldin_score(X_scaled, clusters)

print(f"\n=== Clustering Metrics ===")
print(f"Optimal number of clusters: {optimal_k}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Inertia: {model.inertia_:.2f}")

# Cluster analysis
print(f"\n=== Cluster Analysis ===")
for i in range(optimal_k):
    cluster_data = copyData[copyData['cluster'] == i]
    print(f"Cluster {i}: {len(cluster_data)} customers")
    print(f"  Average age: {cluster_data['age'].mean():.1f}")
    print(f"  Average income: ${cluster_data['annual_income'].mean():,.0f}")
    print(f"  Average spending: {cluster_data['spending_score'].mean():.1f}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: Elbow method
plt.subplot(3, 3, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

# Plot 2: Silhouette scores
plt.subplot(3, 3, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k')
plt.grid(True)

# Plot 3: Age vs Income
plt.subplot(3, 3, 3)
scatter = plt.scatter(copyData['age'], copyData['annual_income'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
centroids_unscaled = scaler.inverse_transform(model.cluster_centers_)
plt.scatter(centroids_unscaled[:, 0], centroids_unscaled[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel('Age')
plt.ylabel('Annual Income ($)')
plt.title('Age vs Annual Income')
plt.legend()
plt.colorbar(scatter)

# Plot 4: Age vs Spending Score
plt.subplot(3, 3, 4)
scatter = plt.scatter(copyData['age'], copyData['spending_score'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.scatter(centroids_unscaled[:, 0], centroids_unscaled[:, 2], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Age vs Spending Score')
plt.legend()
plt.colorbar(scatter)

# Plot 5: Income vs Spending Score
plt.subplot(3, 3, 5)
scatter = plt.scatter(copyData['annual_income'], copyData['spending_score'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.scatter(centroids_unscaled[:, 1], centroids_unscaled[:, 2], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score')
plt.title('Income vs Spending Score')
plt.legend()
plt.colorbar(scatter)

# Plot 6: Cluster distribution
plt.subplot(3, 3, 6)
cluster_counts = copyData['cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Customer Distribution by Cluster')

# CUSTOM INPUT
print("\n=== Custom Customer Classification ===")
print("Enter customer details for segmentation:")
age_input = float(input("Enter customer age (18-70): "))
income_input = float(input("Enter annual income ($15000-$150000): "))
spending_input = float(input("Enter spending score (1-100): "))

custom_data = np.array([[age_input, income_input, spending_input]])
custom_data_scaled = scaler.transform(custom_data)
custom_cluster = model.predict(custom_data_scaled)

print(f"\nCustomer Profile Analysis:")
print(f"Age: {age_input}")
print(f"Annual Income: ${income_input:,.0f}")
print(f"Spending Score: {spending_input}")
print(f"Assigned to Cluster: {custom_cluster[0]}")

# Provide cluster interpretation
cluster_names = {
    0: "Young Low Spenders",
    1: "Middle-aged High Earners", 
    2: "Senior Savers",
    3: "Young High Spenders"
}

if custom_cluster[0] in cluster_names:
    print(f"Cluster Type: {cluster_names[custom_cluster[0]]}")

# Plot 7: Customer profile with input
plt.subplot(3, 3, 7)
scatter = plt.scatter(copyData['annual_income'], copyData['spending_score'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.scatter(centroids_unscaled[:, 1], centroids_unscaled[:, 2], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.scatter(income_input, spending_input, c='black', marker='X', s=300, label='Your Customer', edgecolors='white', linewidths=2)
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score')
plt.title('Your Customer Classification')
plt.legend()
plt.colorbar(scatter)

# Plot 8: Cluster characteristics heatmap
plt.subplot(3, 3, 8)
cluster_means = copyData.groupby('cluster')[['age', 'annual_income', 'spending_score']].mean()
sns.heatmap(cluster_means.T, annot=True, fmt='.1f', cmap='viridis', cbar_kws={'label': 'Average Value'})
plt.title('Cluster Characteristics')
plt.ylabel('Features')

# Plot 9: 3D visualization preview (2D projection)
plt.subplot(3, 3, 9)
# Use PCA for 2D projection of 3D data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=copyData['cluster'], cmap='viridis', alpha=0.7)
centroids_pca = pca.transform(model.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
if len(custom_data) > 0:
    custom_pca = pca.transform(custom_data_scaled)
    plt.scatter(custom_pca[:, 0], custom_pca[:, 1], c='black', marker='X', s=300, label='Your Customer', edgecolors='white', linewidths=2)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Projection of Clusters')
plt.legend()
plt.colorbar(scatter)

plt.tight_layout()
plt.savefig('customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'customer_segmentation_analysis.png'")