import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import random

# DATA LOAD
random.seed(42)
np.random.seed(42)

# Generate synthetic customer data
data = {
    'age': [],
    'annual_income': [],
    'spending_score': []
}

for i in range(500):
    age = np.random.randint(18, 70)
    annual_income = np.random.randint(15000, 150000)
    spending_score = np.random.randint(1, 100)
    
    data['age'].append(age)
    data['annual_income'].append(annual_income)
    data['spending_score'].append(spending_score)

df = pd.DataFrame(data)
copyData = df.copy()

print("=== KMeans Clustering ===")
print(f"Dataset shape: {copyData.shape}")
print(copyData.head())

# MODEL
X = copyData[['age', 'annual_income', 'spending_score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)
copyData['cluster'] = clusters

# METRICS
silhouette_avg = silhouette_score(X_scaled, clusters)
davies_bouldin = davies_bouldin_score(X_scaled, clusters)

print(f"\n=== Clustering Metrics ===")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Number of clusters: {model.n_clusters}")

# PLOTS
plt.figure(figsize=(12, 8))

# Plot 1: Age vs Annual Income
plt.subplot(2, 2, 1)
scatter = plt.scatter(copyData['age'], copyData['annual_income'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
centroids_unscaled = scaler.inverse_transform(model.cluster_centers_)
plt.scatter(centroids_unscaled[:, 0], centroids_unscaled[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.title('KMeans Clustering: Age vs Annual Income')
plt.legend()
plt.colorbar(scatter)

# Plot 2: Age vs Spending Score
plt.subplot(2, 2, 2)
scatter = plt.scatter(copyData['age'], copyData['spending_score'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.scatter(centroids_unscaled[:, 0], centroids_unscaled[:, 2], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('KMeans Clustering: Age vs Spending Score')
plt.legend()
plt.colorbar(scatter)

# Plot 3: Annual Income vs Spending Score
plt.subplot(2, 2, 3)
scatter = plt.scatter(copyData['annual_income'], copyData['spending_score'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.scatter(centroids_unscaled[:, 1], centroids_unscaled[:, 2], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('KMeans Clustering: Income vs Spending')
plt.legend()
plt.colorbar(scatter)

# CUSTOM INPUT
print("\n=== Custom Prediction ===")
print("Enter customer details for clustering:")
age_input = float(input("Enter age (18-70): "))
income_input = float(input("Enter annual income (15000-150000): "))
spending_input = float(input("Enter spending score (1-100): "))

custom_data = np.array([[age_input, income_input, spending_input]])
custom_data_scaled = scaler.transform(custom_data)
custom_cluster = model.predict(custom_data_scaled)

print(f"\nCustomer belongs to cluster: {custom_cluster[0]}")

# Add custom input to plots
plt.subplot(2, 2, 4)
scatter = plt.scatter(copyData['annual_income'], copyData['spending_score'], c=copyData['cluster'], cmap='viridis', alpha=0.7)
plt.scatter(centroids_unscaled[:, 1], centroids_unscaled[:, 2], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.scatter(income_input, spending_input, c='black', marker='X', s=300, label='Your Input', edgecolors='white', linewidths=2)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Your Customer Classification')
plt.legend()
plt.colorbar(scatter)

plt.tight_layout()
plt.savefig('kmeans_clustering.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as 'kmeans_clustering.png'")

