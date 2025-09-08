import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import seaborn as sns
import time

# DATA LOAD
np.random.seed(42)

# Generate high-dimensional synthetic dataset with clear clusters
X_high_dim, y_labels = make_classification(
    n_samples=500, 
    n_features=50, 
    n_informative=30, 
    n_redundant=10,
    n_clusters_per_class=3,
    class_sep=1.5,
    random_state=42
)

# Create DataFrame
feature_names = [f'feature_{i+1}' for i in range(50)]
df = pd.DataFrame(X_high_dim, columns=feature_names)
df['target'] = y_labels
copyData = df.copy()

print("=== t-Distributed Stochastic Neighbor Embedding (t-SNE) ===")
print(f"Dataset shape: {copyData.shape}")
print(copyData.head())
print(f"\nDataset info:")
print(f"Features: {len(feature_names)}")
print(f"Classes: {len(np.unique(y_labels))}")
print(f"Samples per class: {np.bincount(y_labels)}")

# MODEL
X = copyData[feature_names]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pre-process with PCA for computational efficiency (recommended for high-dim data)
print(f"\n=== Preprocessing with PCA ===")
pca_preprocess = PCA(n_components=min(50, X_scaled.shape[1]))
X_pca = pca_preprocess.fit_transform(X_scaled)
explained_variance = np.sum(pca_preprocess.explained_variance_ratio_)
print(f"PCA preprocessing: {X_scaled.shape[1]} → {X_pca.shape[1]} dimensions")
print(f"Retained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")

# t-SNE with different perplexity values
perplexity_values = [5, 15, 30, 50]
tsne_results = {}

print(f"\n=== t-SNE with Different Perplexity Values ===")
for perp in perplexity_values:
    print(f"Computing t-SNE with perplexity={perp}...")
    start_time = time.time()
    
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate=200,
        n_iter=1000,
        random_state=42,
        verbose=0
    )
    
    X_tsne = tsne.fit_transform(X_pca)
    end_time = time.time()
    
    tsne_results[perp] = {
        'model': tsne,
        'embedding': X_tsne,
        'kl_divergence': tsne.kl_divergence_,
        'time': end_time - start_time
    }
    
    print(f"  KL divergence: {tsne.kl_divergence_:.4f}")
    print(f"  Time: {end_time - start_time:.2f} seconds")

# Choose optimal perplexity (typically 30 works well)
optimal_perplexity = 30
best_tsne = tsne_results[optimal_perplexity]['model']
best_embedding = tsne_results[optimal_perplexity]['embedding']

# Create DataFrame with t-SNE results
df_tsne = pd.DataFrame(best_embedding, columns=['tSNE1', 'tSNE2'])
df_tsne['target'] = copyData['target']

# METRICS
print(f"\n=== t-SNE Metrics (perplexity={optimal_perplexity}) ===")
print(f"Final KL divergence: {best_tsne.kl_divergence_:.4f}")
print(f"Number of iterations: {best_tsne.n_iter_}")
print(f"Dimensionality reduction: {X.shape[1]} → 2 dimensions")

# Calculate cluster separation metrics
from sklearn.metrics import silhouette_score
silhouette_tsne = silhouette_score(best_embedding, y_labels)
print(f"Silhouette score on t-SNE embedding: {silhouette_tsne:.4f}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: Perplexity comparison
plt.subplot(3, 3, 1)
perp_list = list(tsne_results.keys())
kl_div_list = [tsne_results[p]['kl_divergence'] for p in perp_list]
time_list = [tsne_results[p]['time'] for p in perp_list]

ax1 = plt.gca()
color = 'tab:red'
ax1.set_xlabel('Perplexity')
ax1.set_ylabel('KL Divergence', color=color)
ax1.plot(perp_list, kl_div_list, 'o-', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Time (seconds)', color=color)
ax2.plot(perp_list, time_list, 's-', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Perplexity vs KL Divergence & Time')

# Plot 2: Main t-SNE result
plt.subplot(3, 3, 2)
scatter = plt.scatter(df_tsne['tSNE1'], df_tsne['tSNE2'], c=df_tsne['target'], cmap='viridis', alpha=0.7)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title(f't-SNE Embedding (perplexity={optimal_perplexity})')
plt.colorbar(scatter)

# Plot 3: Comparison of different perplexities
for i, perp in enumerate([5, 15, 30, 50]):
    plt.subplot(3, 4, 5 + i)
    embedding = tsne_results[perp]['embedding']
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_labels, cmap='viridis', alpha=0.7, s=20)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f'Perplexity={perp}')
    if i == 3:
        plt.colorbar(scatter)

# Plot 5: PCA vs t-SNE comparison
plt.subplot(3, 3, 3)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_labels, cmap='viridis', alpha=0.7)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA 2D Projection')
plt.colorbar(scatter)

# CUSTOM INPUT
print("\n=== Custom Data Point Transformation ===")
print("Enter values for first 5 features to see t-SNE projection:")
custom_input = []
for i in range(5):
    value = float(input(f"Enter feature_{i+1} value (-3 to 3): "))
    custom_input.append(value)

# Pad with zeros for remaining features
custom_input.extend([0] * (len(feature_names) - 5))
custom_data = np.array([custom_input])
custom_data_scaled = scaler.transform(custom_data)

# For t-SNE, we cannot directly transform new points (unlike PCA)
# We need to either:
# 1. Re-run t-SNE with the new point included, or
# 2. Find the nearest neighbors and interpolate
# Here we'll use approach 2 for efficiency

print(f"\nOriginal Data Point (first 5 features):")
for i in range(5):
    print(f"feature_{i+1}: {custom_input[i]:.2f}")

# Find k nearest neighbors in the original high-dimensional space
from sklearn.neighbors import NearestNeighbors
k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_scaled)
distances, indices = nn.kneighbors(custom_data_scaled)

# Interpolate position in t-SNE space based on nearest neighbors
weights = 1 / (distances[0] + 1e-10)  # Add small value to avoid division by zero
weights = weights / np.sum(weights)  # Normalize weights

custom_tsne = np.sum(best_embedding[indices[0]] * weights.reshape(-1, 1), axis=0)

print(f"\nApproximate t-SNE Position:")
print(f"t-SNE 1: {custom_tsne[0]:.4f}")
print(f"t-SNE 2: {custom_tsne[1]:.4f}")
print(f"Based on {k} nearest neighbors")

# Plot 6: Custom point visualization
plt.subplot(3, 3, 6)
scatter = plt.scatter(df_tsne['tSNE1'], df_tsne['tSNE2'], c=df_tsne['target'], cmap='viridis', alpha=0.7)
plt.scatter(custom_tsne[0], custom_tsne[1], c='red', marker='X', s=300, label='Your Point', edgecolors='white', linewidths=2)
# Also show the nearest neighbors
nn_points = best_embedding[indices[0]]
plt.scatter(nn_points[:, 0], nn_points[:, 1], c='orange', marker='o', s=100, label='Nearest Neighbors', alpha=0.8)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Your Point in t-SNE Space')
plt.legend()
plt.colorbar(scatter)

# Plot 7: Feature correlation with t-SNE dimensions
plt.subplot(3, 3, 7)
# Calculate correlation between original features and t-SNE dimensions
correlations_tsne1 = [np.corrcoef(X_scaled[:, i], best_embedding[:, 0])[0, 1] for i in range(min(10, X_scaled.shape[1]))]
correlations_tsne2 = [np.corrcoef(X_scaled[:, i], best_embedding[:, 1])[0, 1] for i in range(min(10, X_scaled.shape[1]))]

x_pos = np.arange(len(correlations_tsne1))
width = 0.35

plt.bar(x_pos - width/2, correlations_tsne1, width, label='t-SNE 1', alpha=0.7)
plt.bar(x_pos + width/2, correlations_tsne2, width, label='t-SNE 2', alpha=0.7)
plt.xlabel('Original Features (first 10)')
plt.ylabel('Correlation')
plt.title('Feature Correlation with t-SNE Dimensions')
plt.xticks(x_pos, [f'F{i+1}' for i in range(len(correlations_tsne1))])
plt.legend()

# Plot 8: Convergence analysis (KL divergence over iterations)
plt.subplot(3, 3, 8)
# Note: sklearn's t-SNE doesn't provide iteration-wise KL divergence
# This is a placeholder showing final KL divergence for different perplexities
perp_list = list(tsne_results.keys())
kl_div_list = [tsne_results[p]['kl_divergence'] for p in perp_list]
plt.plot(perp_list, kl_div_list, 'o-', linewidth=2, markersize=8)
plt.xlabel('Perplexity')
plt.ylabel('Final KL Divergence')
plt.title('KL Divergence vs Perplexity')
plt.grid(True)

# Plot 9: Class distribution in t-SNE space
plt.subplot(3, 3, 9)
unique_classes = np.unique(y_labels)
colors = ['blue', 'orange', 'green', 'red', 'purple']
for i, class_label in enumerate(unique_classes):
    mask = y_labels == class_label
    plt.scatter(best_embedding[mask, 0], best_embedding[mask, 1], 
               c=colors[i % len(colors)], label=f'Class {class_label}', alpha=0.7)

plt.scatter(custom_tsne[0], custom_tsne[1], c='black', marker='X', s=300, label='Your Point', edgecolors='white', linewidths=2)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Class Distribution in t-SNE Space')
plt.legend()

plt.tight_layout()
plt.savefig('tsne_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'tsne_analysis.png'")