import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# DATA LOAD
data = pd.read_csv('image_features_dataset.csv')
copyData = data.copy()

print("=== Image Compression Simulation with PCA ===")
print(f"Dataset shape: {copyData.shape}")
print(copyData.head())
print(f"\nDataset info:")
print(copyData.describe())

# MODEL
# Extract feature columns (excluding metadata)
feature_columns = [col for col in copyData.columns if col.startswith('pixel_')]
X = copyData[feature_columns]
image_ids = copyData['image_id'] if 'image_id' in copyData.columns else range(len(copyData))

print(f"\nOriginal image dimensions: {len(feature_columns)} features (pixels)")
print(f"Number of images: {len(X)}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with different numbers of components to simulate compression levels
compression_ratios = [0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.10]
pca_results = {}

print(f"\n=== PCA Compression Analysis ===")
for ratio in compression_ratios:
    # Find number of components needed for this variance ratio
    pca_temp = PCA()
    pca_temp.fit(X_scaled)
    cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
    n_components = np.argmax(cumsum_var >= ratio) + 1
    
    # Apply PCA with selected number of components
    pca = PCA(n_components=n_components)
    X_compressed = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_compressed)
    
    # Calculate compression metrics
    original_size = X_scaled.shape[1]
    compressed_size = n_components
    compression_ratio_actual = compressed_size / original_size
    
    # Calculate reconstruction error
    mse = np.mean((X_scaled - X_reconstructed)**2)
    
    pca_results[ratio] = {
        'pca': pca,
        'n_components': n_components,
        'compressed_data': X_compressed,
        'reconstructed_data': X_reconstructed,
        'compression_ratio': compression_ratio_actual,
        'mse': mse,
        'variance_explained': np.sum(pca.explained_variance_ratio_)
    }
    
    print(f"Target variance: {ratio:.0%}")
    print(f"  Components: {n_components}/{original_size} ({compression_ratio_actual:.1%})")
    print(f"  Actual variance explained: {pca_results[ratio]['variance_explained']:.4f}")
    print(f"  Reconstruction MSE: {mse:.6f}")

# Choose optimal compression (90% variance retained)
optimal_ratio = 0.90
optimal_pca = pca_results[optimal_ratio]['pca']
optimal_compressed = pca_results[optimal_ratio]['compressed_data']
optimal_reconstructed = pca_results[optimal_ratio]['reconstructed_data']

# METRICS
print(f"\n=== Optimal Compression Metrics (90% variance) ===")
print(f"Original dimensions: {X_scaled.shape[1]}")
print(f"Compressed dimensions: {optimal_pca.n_components}")
print(f"Compression ratio: {pca_results[optimal_ratio]['compression_ratio']:.1%}")
print(f"Variance explained: {pca_results[optimal_ratio]['variance_explained']:.4f}")
print(f"Reconstruction error (MSE): {pca_results[optimal_ratio]['mse']:.6f}")

# Size reduction calculation
original_size_mb = (X_scaled.shape[0] * X_scaled.shape[1] * 4) / (1024 * 1024)  # 4 bytes per float
compressed_size_mb = (X_scaled.shape[0] * optimal_pca.n_components * 4) / (1024 * 1024)
size_reduction = (1 - compressed_size_mb / original_size_mb) * 100

print(f"\nStorage analysis:")
print(f"Original size: {original_size_mb:.2f} MB")
print(f"Compressed size: {compressed_size_mb:.2f} MB")
print(f"Size reduction: {size_reduction:.1f}%")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: Variance explained vs number of components
plt.subplot(3, 3, 1)
pca_full = PCA()
pca_full.fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', linewidth=2)
plt.axhline(y=0.90, color='red', linestyle='--', label='90% threshold')
plt.axhline(y=0.95, color='orange', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Explained')
plt.legend()
plt.grid(True)

# Plot 2: Compression ratio vs reconstruction error
plt.subplot(3, 3, 2)
ratios = list(pca_results.keys())
mse_values = [pca_results[r]['mse'] for r in ratios]
compression_ratios_actual = [pca_results[r]['compression_ratio'] for r in ratios]

plt.scatter(compression_ratios_actual, mse_values, c='red', s=100)
for i, ratio in enumerate(ratios):
    plt.annotate(f'{ratio:.0%}', (compression_ratios_actual[i], mse_values[i]), 
                xytext=(5, 5), textcoords='offset points')
plt.xlabel('Compression Ratio')
plt.ylabel('Reconstruction MSE')
plt.title('Compression vs Quality Trade-off')
plt.grid(True)

# Plot 3: First few principal components visualization
plt.subplot(3, 3, 3)
n_show = min(10, optimal_pca.n_components)
components_to_show = range(n_show)
variances_to_show = optimal_pca.explained_variance_ratio_[:n_show]

plt.bar(components_to_show, variances_to_show, alpha=0.7, color='blue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title(f'Top {n_show} Components Variance')
plt.xticks(components_to_show)

# Plot 4: Original vs Compressed data comparison (first image)
plt.subplot(3, 3, 4)
image_idx = 0
original_image = X_scaled[image_idx]
reconstructed_image = optimal_reconstructed[image_idx]

# Assuming square images, try to reshape for visualization
img_size = int(np.sqrt(len(original_image)))
if img_size * img_size == len(original_image):
    original_2d = original_image.reshape(img_size, img_size)
    plt.imshow(original_2d, cmap='gray')
    plt.title(f'Original Image {image_idx+1}')
    plt.axis('off')
else:
    # If not square, plot as 1D signal
    plt.plot(original_image[:100], label='Original', alpha=0.7)
    plt.plot(reconstructed_image[:100], label='Reconstructed', alpha=0.7)
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Value')
    plt.title('Original vs Reconstructed (first 100 pixels)')
    plt.legend()

# Plot 5: Reconstructed image
plt.subplot(3, 3, 5)
if img_size * img_size == len(original_image):
    reconstructed_2d = reconstructed_image.reshape(img_size, img_size)
    plt.imshow(reconstructed_2d, cmap='gray')
    plt.title(f'Reconstructed Image {image_idx+1}')
    plt.axis('off')
else:
    plt.plot(reconstructed_image[:100], color='red', alpha=0.7)
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Value')
    plt.title('Reconstructed Signal (first 100 pixels)')

# Plot 6: Reconstruction error distribution
plt.subplot(3, 3, 6)
reconstruction_errors = np.mean((X_scaled - optimal_reconstructed)**2, axis=1)
plt.hist(reconstruction_errors, bins=30, alpha=0.7, color='green')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Errors')

# CUSTOM INPUT
print("\n=== Custom Image Compression ===")
print("Enter a custom image row index to analyze:")
try:
    custom_idx = int(input(f"Enter image index (0-{len(X)-1}): "))
    if 0 <= custom_idx < len(X):
        custom_image = X_scaled[custom_idx]
        custom_compressed = optimal_pca.transform(custom_image.reshape(1, -1))
        custom_reconstructed = optimal_pca.inverse_transform(custom_compressed)
        custom_error = np.mean((custom_image - custom_reconstructed[0])**2)
        
        print(f"\nCustom Image Analysis:")
        print(f"Image index: {custom_idx}")
        print(f"Original dimensions: {len(custom_image)}")
        print(f"Compressed dimensions: {len(custom_compressed[0])}")
        print(f"Reconstruction error: {custom_error:.6f}")
        print(f"Compression ratio: {len(custom_compressed[0])/len(custom_image):.1%}")
        
        # Compare with average error
        avg_error = np.mean(reconstruction_errors)
        if custom_error < avg_error:
            print("✅ This image compresses better than average!")
        else:
            print("⚠️ This image has higher compression error than average.")
    else:
        custom_idx = 0
        print(f"Invalid index, using image {custom_idx}")
except:
    custom_idx = 0
    print(f"Using default image {custom_idx}")

# Plot 7: Custom image analysis
plt.subplot(3, 3, 7)
custom_image = X_scaled[custom_idx]
custom_reconstructed_single = optimal_reconstructed[custom_idx]

plt.plot(custom_image[:50], 'b-', label='Original', linewidth=2)
plt.plot(custom_reconstructed_single[:50], 'r--', label='Reconstructed', linewidth=2)
plt.xlabel('Pixel Index')
plt.ylabel('Pixel Value')
plt.title(f'Your Image {custom_idx+1} (first 50 pixels)')
plt.legend()

# Plot 8: Component weights heatmap
plt.subplot(3, 3, 8)
# Show first few components as heatmap
n_components_show = min(8, optimal_pca.n_components)
components_subset = optimal_pca.components_[:n_components_show, :min(50, X.shape[1])]
sns.heatmap(components_subset, cmap='RdBu_r', center=0, 
            yticklabels=[f'PC{i+1}' for i in range(n_components_show)],
            xticklabels=False)
plt.title('Principal Components Visualization')
plt.ylabel('Components')

# Plot 9: Storage savings analysis
plt.subplot(3, 3, 9)
ratios = list(pca_results.keys())
storage_savings = [(1 - pca_results[r]['compression_ratio']) * 100 for r in ratios]
variance_retained = [pca_results[r]['variance_explained'] * 100 for r in ratios]

plt.scatter(variance_retained, storage_savings, c='purple', s=100)
for i, ratio in enumerate(ratios):
    plt.annotate(f'{ratio:.0%}', (variance_retained[i], storage_savings[i]), 
                xytext=(5, 5), textcoords='offset points')
plt.xlabel('Variance Retained (%)')
plt.ylabel('Storage Savings (%)')
plt.title('Storage Savings vs Quality')
plt.grid(True)

plt.tight_layout()
plt.savefig('image_compression_pca.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'image_compression_pca.png'")