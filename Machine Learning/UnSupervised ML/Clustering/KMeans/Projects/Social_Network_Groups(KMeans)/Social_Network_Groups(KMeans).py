import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# DATA LOAD
data = pd.read_csv('social_network_dataset.csv')
copyData = data.copy()

print("=== Social Network User Clustering with KMeans ===")
print(f"Dataset shape: {copyData.shape}")
print(copyData.head())
print(f"\nDataset info:")
print(copyData.describe())

# MODEL
X = copyData[['follower_count', 'following_count', 'posts_count', 'likes_avg', 'activity_score']]
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

# Choose optimal k (k=5 based on analysis)
optimal_k = 5
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
print(f"\n=== User Group Analysis ===")
group_names = {
    0: "Casual Users",
    1: "Influencers",
    2: "Active Enthusiasts", 
    3: "Lurkers",
    4: "Content Creators"
}

for i in range(optimal_k):
    cluster_data = copyData[copyData['cluster'] == i]
    print(f"Cluster {i} ({group_names.get(i, 'Unknown')}): {len(cluster_data)} users")
    print(f"  Avg followers: {cluster_data['follower_count'].mean():.0f}")
    print(f"  Avg following: {cluster_data['following_count'].mean():.0f}")
    print(f"  Avg posts: {cluster_data['posts_count'].mean():.0f}")
    print(f"  Avg likes per post: {cluster_data['likes_avg'].mean():.1f}")
    print(f"  Avg activity score: {cluster_data['activity_score'].mean():.1f}")

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

# Plot 3: Followers vs Following
plt.subplot(3, 3, 3)
scatter = plt.scatter(copyData['follower_count'], copyData['following_count'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
centroids_unscaled = scaler.inverse_transform(model.cluster_centers_)
plt.scatter(centroids_unscaled[:, 0], centroids_unscaled[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel('Follower Count')
plt.ylabel('Following Count')
plt.title('Followers vs Following')
plt.legend()
plt.colorbar(scatter)

# Plot 4: Posts vs Likes
plt.subplot(3, 3, 4)
scatter = plt.scatter(copyData['posts_count'], copyData['likes_avg'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.scatter(centroids_unscaled[:, 2], centroids_unscaled[:, 3], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel('Posts Count')
plt.ylabel('Average Likes per Post')
plt.title('Posts vs Average Likes')
plt.legend()
plt.colorbar(scatter)

# Plot 5: Followers vs Activity Score
plt.subplot(3, 3, 5)
scatter = plt.scatter(copyData['follower_count'], copyData['activity_score'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.scatter(centroids_unscaled[:, 0], centroids_unscaled[:, 4], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel('Follower Count')
plt.ylabel('Activity Score')
plt.title('Followers vs Activity Score')
plt.legend()
plt.colorbar(scatter)

# Plot 6: Cluster distribution
plt.subplot(3, 3, 6)
cluster_counts = copyData['cluster'].value_counts().sort_index()
colors = ['blue', 'orange', 'green', 'red', 'purple']
bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors[:len(cluster_counts)])
plt.xlabel('Cluster')
plt.ylabel('Number of Users')
plt.title('User Distribution by Cluster')

# Add cluster names to bars
for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{group_names.get(i, "Unknown")}\n({count})', 
             ha='center', va='bottom', fontsize=8)

# CUSTOM INPUT
print("\n=== Custom User Classification ===")
print("Enter user profile for group classification:")
followers = float(input("Enter follower count (0-50000): "))
following = float(input("Enter following count (0-5000): "))
posts = float(input("Enter posts count (0-10000): "))
likes_avg = float(input("Enter average likes per post (0-1000): "))
activity = float(input("Enter activity score (0-100): "))

custom_data = np.array([[followers, following, posts, likes_avg, activity]])
custom_data_scaled = scaler.transform(custom_data)
custom_cluster = model.predict(custom_data_scaled)

print(f"\nUser Profile Analysis:")
print(f"Followers: {followers:.0f}")
print(f"Following: {following:.0f}")
print(f"Posts: {posts:.0f}")
print(f"Avg Likes per Post: {likes_avg:.1f}")
print(f"Activity Score: {activity:.1f}")
print(f"Assigned to Cluster: {custom_cluster[0]}")
print(f"User Group: {group_names.get(custom_cluster[0], 'Unknown')}")

# Plot 7: User profile with input
plt.subplot(3, 3, 7)
scatter = plt.scatter(copyData['follower_count'], copyData['activity_score'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.scatter(centroids_unscaled[:, 0], centroids_unscaled[:, 4], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.scatter(followers, activity, c='black', marker='X', s=300, label='Your Profile', edgecolors='white', linewidths=2)
plt.xlabel('Follower Count')
plt.ylabel('Activity Score')
plt.title('Your User Classification')
plt.legend()
plt.colorbar(scatter)

# Plot 8: Cluster characteristics radar chart simulation
plt.subplot(3, 3, 8)
cluster_means = copyData.groupby('cluster')[['follower_count', 'following_count', 'posts_count', 'likes_avg', 'activity_score']].mean()
# Normalize for better visualization
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
sns.heatmap(cluster_means_norm.T, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'Normalized Value'})
plt.title('Cluster Characteristics')
plt.ylabel('Features')

# Plot 9: 3D visualization using PCA
plt.subplot(3, 3, 9)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=copyData['cluster'], cmap='viridis', alpha=0.7)
centroids_pca = pca.transform(model.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
if len(custom_data) > 0:
    custom_pca = pca.transform(custom_data_scaled)
    plt.scatter(custom_pca[:, 0], custom_pca[:, 1], c='black', marker='X', s=300, label='Your Profile', edgecolors='white', linewidths=2)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA Projection of User Groups')
plt.legend()
plt.colorbar(scatter)

plt.tight_layout()
plt.savefig('social_network_clustering.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'social_network_clustering.png'")