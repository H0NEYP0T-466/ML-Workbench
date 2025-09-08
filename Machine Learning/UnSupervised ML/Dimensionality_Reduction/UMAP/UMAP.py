import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

# DATA LOAD
np.random.seed(42)

# Generate high-dimensional synthetic dataset with clear clusters
X_high_dim, y_labels = make_classification(
    n_samples=600, 
    n_features=30, 
    n_informative=20, 
    n_redundant=5,
    n_clusters_per_class=2,
    class_sep=1.5,
    random_state=42
)

# Create DataFrame
feature_names = [f'feature_{i+1}' for i in range(30)]
df = pd.DataFrame(X_high_dim, columns=feature_names)
df['target'] = y_labels
copyData = df.copy()

print("=== Uniform Manifold Approximation and Projection (UMAP) ===")
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

print(f"\n=== UMAP Analysis ===")

# UMAP with different parameters
umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)

X_umap_2d = umap_2d.fit_transform(X_scaled)
X_umap_3d = umap_3d.fit_transform(X_scaled)

print(f"Original dimensions: {X_scaled.shape[1]}")
print(f"UMAP 2D shape: {X_umap_2d.shape}")
print(f"UMAP 3D shape: {X_umap_3d.shape}")

# Compare with PCA and t-SNE
pca_2d = PCA(n_components=2, random_state=42)
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)

X_pca_2d = pca_2d.fit_transform(X_scaled)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)

print(f"\n=== Method Comparison ===")
print(f"PCA explained variance: {sum(pca_2d.explained_variance_ratio_):.4f}")
print(f"UMAP preserves local structure while reducing global dimensionality")
print(f"t-SNE optimizes for local neighborhood preservation")

# CUSTOM INPUT
print("\n=== Custom Data Point Transformation ===")
print("Enter values for first 5 features to see UMAP projection:")
custom_input = []
try:
    for i in range(5):
        while True:
            try:
                value = float(input(f"Feature {i+1} ({feature_names[i]}): "))
                custom_input.append(value)
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Pad with zeros or mean values for remaining features
    remaining_features = len(feature_names) - 5
    feature_means = X_scaled.mean(axis=0)[5:]
    custom_input.extend(feature_means)
    
    # Transform custom input
    custom_input_array = np.array(custom_input).reshape(1, -1)
    custom_umap_2d = umap_2d.transform(custom_input_array)
    custom_umap_3d = umap_3d.transform(custom_input_array)
    
    print(f"\nYour input in UMAP 2D space: ({custom_umap_2d[0, 0]:.3f}, {custom_umap_2d[0, 1]:.3f})")
    print(f"Your input in UMAP 3D space: ({custom_umap_3d[0, 0]:.3f}, {custom_umap_3d[0, 1]:.3f}, {custom_umap_3d[0, 2]:.3f})")
    
    # Find nearest neighbors in UMAP space
    nn_model = NearestNeighbors(n_neighbors=5)
    nn_model.fit(X_umap_2d)
    distances, indices = nn_model.kneighbors(custom_umap_2d)
    
    print(f"\n=== Nearest Neighbors in UMAP Space ===")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"{i+1}. Sample {idx}: distance={dist:.3f}, class={y_labels[idx]}")
    
except (KeyboardInterrupt, EOFError):
    print("\nUsing default values for visualization...")
    custom_input = list(X_scaled.mean(axis=0))
    custom_input_array = np.array(custom_input).reshape(1, -1)
    custom_umap_2d = umap_2d.transform(custom_input_array)
    custom_umap_3d = umap_3d.transform(custom_input_array)
    
    nn_model = NearestNeighbors(n_neighbors=5)
    nn_model.fit(X_umap_2d)
    distances, indices = nn_model.kneighbors(custom_umap_2d)

# METRICS
print(f"\n=== UMAP Metrics ===")
print(f"UMAP 2D embedding variance: {np.var(X_umap_2d, axis=0)}")
print(f"UMAP 3D embedding variance: {np.var(X_umap_3d, axis=0)}")

# Calculate local structure preservation (approximate)
from scipy.spatial.distance import pdist, squareform
original_distances = squareform(pdist(X_scaled[:100]))  # Sample for efficiency
umap_distances = squareform(pdist(X_umap_2d[:100]))

# Correlation between distance matrices (local structure preservation)
correlation = np.corrcoef(original_distances.flatten(), umap_distances.flatten())[0, 1]
print(f"Local structure preservation (correlation): {correlation:.4f}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: UMAP 2D projection
plt.subplot(3, 3, 1)
scatter = plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], c=y_labels, cmap='viridis', alpha=0.7)
plt.scatter(custom_umap_2d[0, 0], custom_umap_2d[0, 1], 
           c='black', marker='x', s=200, linewidth=3, label='Your Input')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP 2D Projection')
plt.legend()
plt.colorbar(scatter)

# Plot 2: PCA vs UMAP comparison
plt.subplot(3, 3, 2)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_labels, cmap='viridis', alpha=0.7)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA 2D Projection')
plt.colorbar()

# Plot 3: t-SNE vs UMAP comparison
plt.subplot(3, 3, 3)
plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y_labels, cmap='viridis', alpha=0.7)
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.title('t-SNE 2D Projection')
plt.colorbar()

# Plot 4: UMAP with different n_neighbors
plt.subplot(3, 3, 4)
umap_nn5 = umap.UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.1)
X_umap_nn5 = umap_nn5.fit_transform(X_scaled)
plt.scatter(X_umap_nn5[:, 0], X_umap_nn5[:, 1], c=y_labels, cmap='viridis', alpha=0.7)
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP (n_neighbors=5)')
plt.colorbar()

# Plot 5: UMAP with different min_dist
plt.subplot(3, 3, 5)
umap_md05 = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.5)
X_umap_md05 = umap_md05.fit_transform(X_scaled)
plt.scatter(X_umap_md05[:, 0], X_umap_md05[:, 1], c=y_labels, cmap='viridis', alpha=0.7)
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP (min_dist=0.5)')
plt.colorbar()

# Plot 6: Nearest neighbors visualization
plt.subplot(3, 3, 6)
plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], c='lightgray', alpha=0.5)
plt.scatter(X_umap_2d[indices[0], 0], X_umap_2d[indices[0], 1], 
           c=['red', 'orange', 'yellow', 'green', 'blue'], s=100, alpha=0.8)
plt.scatter(custom_umap_2d[0, 0], custom_umap_2d[0, 1], 
           c='black', marker='x', s=200, linewidth=3)
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('Nearest Neighbors to Your Input')

# Plot 7: Class distribution in UMAP space
plt.subplot(3, 3, 7)
unique_classes = np.unique(y_labels)
colors = ['blue', 'orange', 'green', 'red', 'purple']
for i, class_label in enumerate(unique_classes):
    mask = y_labels == class_label
    plt.scatter(X_umap_2d[mask, 0], X_umap_2d[mask, 1], 
               c=colors[i % len(colors)], label=f'Class {class_label}', alpha=0.7)
plt.scatter(custom_umap_2d[0, 0], custom_umap_2d[0, 1], 
           c='black', marker='x', s=200, linewidth=3, label='Your Input')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('Class Distribution in UMAP Space')
plt.legend()

# Plot 8: Parameter comparison
plt.subplot(3, 3, 8)
nn_values = [5, 15, 30, 50]
variance_x = []
variance_y = []
for nn in nn_values:
    umap_temp = umap.UMAP(n_components=2, random_state=42, n_neighbors=nn, min_dist=0.1)
    X_temp = umap_temp.fit_transform(X_scaled)
    variance_x.append(np.var(X_temp[:, 0]))
    variance_y.append(np.var(X_temp[:, 1]))

plt.plot(nn_values, variance_x, 'o-', label='UMAP1 Variance', linewidth=2)
plt.plot(nn_values, variance_y, 's-', label='UMAP2 Variance', linewidth=2)
plt.xlabel('n_neighbors')
plt.ylabel('Variance')
plt.title('UMAP Parameter Sensitivity')
plt.legend()
plt.grid(True)

# Plot 9: 3D UMAP visualization
if X_umap_3d.shape[1] >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(3, 3, 9, projection='3d')
    scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], 
                        c=y_labels, cmap='viridis', alpha=0.6)
    ax.scatter(custom_umap_3d[0, 0], custom_umap_3d[0, 1], custom_umap_3d[0, 2], 
              c='black', marker='x', s=200, linewidth=3)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')
    ax.set_title('UMAP 3D Projection')
    plt.colorbar(scatter)

plt.tight_layout()
plt.savefig('umap_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'umap_analysis.png'")