import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import seaborn as sns

# DATA LOAD
np.random.seed(42)

# Generate high-dimensional synthetic dataset
X_high_dim, y_labels = make_classification(
    n_samples=600, 
    n_features=10, 
    n_informative=8, 
    n_redundant=2, 
    n_clusters_per_class=2,
    random_state=42
)

# Create DataFrame
feature_names = [f'feature_{i+1}' for i in range(10)]
df = pd.DataFrame(X_high_dim, columns=feature_names)
df['target'] = y_labels
copyData = df.copy()

print("=== Principal Component Analysis (PCA) ===")
print(f"Dataset shape: {copyData.shape}")
print(copyData.head())
print(f"\nDataset info:")
print(copyData.describe())

# MODEL
X = copyData[feature_names]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA with all components first to see explained variance
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

print(f"\n=== PCA Analysis ===")
print("Explained variance ratio for each component:")
for i, var_ratio in enumerate(pca_full.explained_variance_ratio_):
    print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")

print(f"\nCumulative explained variance:")
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
for i, cumvar in enumerate(cumsum_var):
    print(f"PC1-PC{i+1}: {cumvar:.4f} ({cumvar*100:.2f}%)")

# Find number of components for 95% variance
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")

# Apply PCA with optimal number of components
optimal_components = min(3, n_components_95)  # Use 3 for visualization
pca = PCA(n_components=optimal_components)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame with PCA results
pca_columns = [f'PC{i+1}' for i in range(optimal_components)]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)
df_pca['target'] = copyData['target']

# METRICS
total_variance_explained = np.sum(pca.explained_variance_ratio_)
print(f"\n=== PCA Metrics ===")
print(f"Number of components: {optimal_components}")
print(f"Total variance explained: {total_variance_explained:.4f} ({total_variance_explained*100:.2f}%)")
print(f"Dimensionality reduction: {X.shape[1]} → {optimal_components} dimensions")

# Component loadings (feature contributions)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(loadings, columns=pca_columns, index=feature_names)

print(f"\n=== Feature Importance in Principal Components ===")
for i, pc in enumerate(pca_columns):
    print(f"\n{pc} (explains {pca.explained_variance_ratio_[i]*100:.2f}% variance):")
    sorted_features = loadings_df[pc].abs().sort_values(ascending=False)
    for feature, loading in sorted_features.head(3).items():
        print(f"  {feature}: {loading:.4f}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: Explained variance ratio
plt.subplot(3, 3, 1)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), pca_full.explained_variance_ratio_, alpha=0.7, color='blue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Component')
plt.xticks(range(1, len(pca_full.explained_variance_ratio_) + 1))

# Plot 2: Cumulative explained variance
plt.subplot(3, 3, 2)
plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2)
plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True)

# Plot 3: PCA 2D visualization
plt.subplot(3, 3, 3)
scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['target'], cmap='viridis', alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA 2D Projection')
plt.colorbar(scatter)

# Plot 4: Feature loadings heatmap
plt.subplot(3, 3, 4)
sns.heatmap(loadings_df.iloc[:, :min(3, optimal_components)], annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Feature Loadings Matrix')
plt.ylabel('Original Features')

# Plot 5: Biplot (if 2D)
if optimal_components >= 2:
    plt.subplot(3, 3, 5)
    # Plot data points
    scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['target'], cmap='viridis', alpha=0.6)
    
    # Plot feature vectors
    scale_factor = 3
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings_df.loc[feature, 'PC1'] * scale_factor, 
                 loadings_df.loc[feature, 'PC2'] * scale_factor,
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        plt.text(loadings_df.loc[feature, 'PC1'] * scale_factor * 1.1,
                loadings_df.loc[feature, 'PC2'] * scale_factor * 1.1,
                feature, fontsize=8, ha='center', va='center')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title('PCA Biplot')
    plt.colorbar(scatter)

# Plot 6: Original vs Reconstructed data comparison
plt.subplot(3, 3, 6)
# Reconstruct data from PCA
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean((X_scaled - X_reconstructed)**2, axis=1)
plt.hist(reconstruction_error, bins=30, alpha=0.7, color='green')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('PCA Reconstruction Error Distribution')

# CUSTOM INPUT
print("\n=== Custom Data Point Transformation ===")
print("Enter values for features to see PCA transformation:")
custom_input = []
for i, feature in enumerate(feature_names):
    value = float(input(f"Enter {feature} value (-3 to 3): "))
    custom_input.append(value)

custom_data = np.array([custom_input])
custom_data_scaled = scaler.transform(custom_data)
custom_pca = pca.transform(custom_data_scaled)

print(f"\nOriginal Data Point:")
for i, (feature, value) in enumerate(zip(feature_names, custom_input)):
    print(f"{feature}: {value:.2f}")

print(f"\nPCA Transformed Point:")
for i, pc_value in enumerate(custom_pca[0]):
    print(f"PC{i+1}: {pc_value:.4f}")

# Reconstruct the point
custom_reconstructed = pca.inverse_transform(custom_pca)
custom_reconstructed_original = scaler.inverse_transform(custom_reconstructed)

reconstruction_error = np.mean((custom_data_scaled - custom_reconstructed)**2)
print(f"\nReconstruction Error: {reconstruction_error:.6f}")

# Plot 7: Custom point in PCA space
plt.subplot(3, 3, 7)
scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['target'], cmap='viridis', alpha=0.7)
plt.scatter(custom_pca[0, 0], custom_pca[0, 1], c='red', marker='X', s=300, label='Your Point', edgecolors='white', linewidths=2)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('Your Point in PCA Space')
plt.legend()
plt.colorbar(scatter)

# Plot 8: Feature contribution analysis
plt.subplot(3, 3, 8)
feature_importance = np.abs(loadings_df).sum(axis=1).sort_values(ascending=True)
plt.barh(range(len(feature_importance)), feature_importance.values, color='orange', alpha=0.7)
plt.yticks(range(len(feature_importance)), feature_importance.index)
plt.xlabel('Total Absolute Loading')
plt.title('Feature Importance in PCA')

# Plot 9: 3D visualization (if 3 components)
if optimal_components >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(3, 3, 9, projection='3d')
    scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'], c=df_pca['target'], cmap='viridis', alpha=0.7)
    if len(custom_pca[0]) >= 3:
        ax.scatter(custom_pca[0, 0], custom_pca[0, 1], custom_pca[0, 2], c='red', marker='X', s=300, label='Your Point')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax.set_title('3D PCA Visualization')
    plt.colorbar(scatter)
else:
    # Alternative plot if no 3rd component
    plt.subplot(3, 3, 9)
    correlation_matrix = np.corrcoef(X_scaled.T)
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Original Feature Correlation')

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'pca_analysis.png'")