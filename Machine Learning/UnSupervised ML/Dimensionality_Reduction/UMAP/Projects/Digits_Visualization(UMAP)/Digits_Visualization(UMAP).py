import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

# DATA LOAD
np.random.seed(42)

# Load digits dataset (8x8 pixel images of digits 0-9)
digits = load_digits()
X_digits = digits.data  # 64 features (8x8 pixels)
y_digits = digits.target  # digit labels (0-9)

# Create DataFrame
feature_names = [f'pixel_{i+1}' for i in range(64)]
df = pd.DataFrame(X_digits, columns=feature_names)
df['digit'] = y_digits
copyData = df.copy()

print("=== UMAP for Digits Visualization ===")
print(f"Dataset shape: {copyData.shape}")
print(f"Features: 64 pixels (8x8 image)")
print(f"Digits: {sorted(np.unique(y_digits))}")
print(f"Samples per digit: {np.bincount(y_digits)}")
print(f"Total samples: {len(copyData)}")

# MODEL
X = copyData[feature_names]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n=== UMAP Digits Analysis ===")

# UMAP with optimized parameters for digits
umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.1, metric='euclidean')
umap_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=10, min_dist=0.1, metric='euclidean')

X_umap_2d = umap_2d.fit_transform(X_scaled)
X_umap_3d = umap_3d.fit_transform(X_scaled)

print(f"Original dimensions: {X_scaled.shape[1]} (8x8 pixels)")
print(f"UMAP 2D shape: {X_umap_2d.shape}")
print(f"UMAP 3D shape: {X_umap_3d.shape}")

# Compare with PCA and t-SNE
pca_2d = PCA(n_components=2, random_state=42)
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)

X_pca_2d = pca_2d.fit_transform(X_scaled)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)

print(f"\n=== Method Comparison for Digits ===")
print(f"PCA explained variance: {sum(pca_2d.explained_variance_ratio_):.4f}")
print(f"UMAP preserves digit structure and separates classes")
print(f"t-SNE creates tight clusters for each digit")

# CUSTOM INPUT
print("\n=== Custom Digit Drawing ===")
print("Enter pixel values (0-16) for an 8x8 digit image:")
print("You can draw a simple digit by entering values row by row")
print("(Higher values = darker pixels, 0 = white, 16 = black)")

custom_digit = []
try:
    print("\nEnter 8 rows of 8 values each (space-separated):")
    for row in range(8):
        while True:
            try:
                row_input = input(f"Row {row+1}: ").strip()
                if row_input.lower() in ['skip', 'default', '']:
                    # Use a default digit pattern (simple "1")
                    default_pattern = [
                        [0, 0, 4, 8, 4, 0, 0, 0],
                        [0, 0, 8, 8, 4, 0, 0, 0],
                        [0, 0, 4, 8, 4, 0, 0, 0],
                        [0, 0, 4, 8, 4, 0, 0, 0],
                        [0, 0, 4, 8, 4, 0, 0, 0],
                        [0, 0, 4, 8, 4, 0, 0, 0],
                        [0, 0, 4, 8, 4, 0, 0, 0],
                        [0, 0, 12, 16, 12, 0, 0, 0]
                    ]
                    custom_digit = [val for row in default_pattern for val in row]
                    print("Using default digit pattern (resembles '1')")
                    break
                
                values = [float(x) for x in row_input.split()]
                if len(values) != 8:
                    print("Please enter exactly 8 values separated by spaces")
                    continue
                if any(v < 0 or v > 16 for v in values):
                    print("Values should be between 0 and 16")
                    continue
                custom_digit.extend(values)
                break
            except ValueError:
                print("Please enter valid numbers or 'skip' for default")
    
    if len(custom_digit) == 64:
        # Transform custom digit
        custom_digit_array = np.array(custom_digit).reshape(1, -1)
        custom_digit_scaled = scaler.transform(custom_digit_array)
        custom_umap_2d = umap_2d.transform(custom_digit_scaled)
        custom_umap_3d = umap_3d.transform(custom_digit_scaled)
        
        print(f"\nYour digit in UMAP 2D space: ({custom_umap_2d[0, 0]:.3f}, {custom_umap_2d[0, 1]:.3f})")
        
        # Find nearest neighbors
        nn_model = NearestNeighbors(n_neighbors=5)
        nn_model.fit(X_umap_2d)
        distances, indices = nn_model.kneighbors(custom_umap_2d)
        
        print(f"\n=== Nearest Digits in UMAP Space ===")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            print(f"{i+1}. Sample {idx}: distance={dist:.3f}, digit={y_digits[idx]}")
        
        # Predict the most likely digit
        nearest_digits = y_digits[indices[0]]
        predicted_digit = np.bincount(nearest_digits).argmax()
        print(f"\nPredicted digit based on nearest neighbors: {predicted_digit}")
    
except KeyboardInterrupt:
    print("\nUsing sample digit for visualization...")
    # Use the first digit from the dataset
    custom_digit = X_digits[0]
    custom_digit_array = custom_digit.reshape(1, -1)
    custom_digit_scaled = scaler.transform(custom_digit_array)
    custom_umap_2d = umap_2d.transform(custom_digit_scaled)
    custom_umap_3d = umap_3d.transform(custom_digit_scaled)
    
    nn_model = NearestNeighbors(n_neighbors=5)
    nn_model.fit(X_umap_2d)
    distances, indices = nn_model.kneighbors(custom_umap_2d)

# METRICS
print(f"\n=== UMAP Digits Metrics ===")
print(f"UMAP 2D embedding variance: {np.var(X_umap_2d, axis=0)}")

# Calculate silhouette score for digit separation
from sklearn.metrics import silhouette_score
silhouette_umap = silhouette_score(X_umap_2d, y_digits)
silhouette_pca = silhouette_score(X_pca_2d, y_digits)
silhouette_tsne = silhouette_score(X_tsne_2d, y_digits)

print(f"Silhouette scores (higher = better digit separation):")
print(f"  UMAP: {silhouette_umap:.4f}")
print(f"  PCA:  {silhouette_pca:.4f}")
print(f"  t-SNE: {silhouette_tsne:.4f}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: UMAP 2D digit visualization
plt.subplot(3, 3, 1)
scatter = plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], c=y_digits, cmap='tab10', alpha=0.7)
plt.scatter(custom_umap_2d[0, 0], custom_umap_2d[0, 1], 
           c='black', marker='x', s=200, linewidth=3, label='Your Digit')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP: Digits Visualization')
plt.legend()
plt.colorbar(scatter, ticks=range(10))

# Plot 2: PCA comparison
plt.subplot(3, 3, 2)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_digits, cmap='tab10', alpha=0.7)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA: Digits Visualization')
plt.colorbar(ticks=range(10))

# Plot 3: t-SNE comparison
plt.subplot(3, 3, 3)
plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y_digits, cmap='tab10', alpha=0.7)
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.title('t-SNE: Digits Visualization')
plt.colorbar(ticks=range(10))

# Plot 4: Custom digit visualization
plt.subplot(3, 3, 4)
digit_image = np.array(custom_digit).reshape(8, 8)
plt.imshow(digit_image, cmap='gray', interpolation='nearest')
plt.title('Your Custom Digit')
plt.axis('off')

# Plot 5: Nearest neighbors
plt.subplot(3, 3, 5)
plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], c='lightgray', alpha=0.3)
colors = ['red', 'orange', 'yellow', 'green', 'blue']
for i, idx in enumerate(indices[0]):
    plt.scatter(X_umap_2d[idx, 0], X_umap_2d[idx, 1], 
               c=colors[i], s=100, alpha=0.8, label=f'NN{i+1}: {y_digits[idx]}')
plt.scatter(custom_umap_2d[0, 0], custom_umap_2d[0, 1], 
           c='black', marker='x', s=200, linewidth=3, label='Your Digit')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('Nearest Neighbors to Your Digit')
plt.legend()

# Plot 6: Digit class centroids
plt.subplot(3, 3, 6)
centroids = []
for digit in range(10):
    digit_mask = y_digits == digit
    centroid = X_umap_2d[digit_mask].mean(axis=0)
    centroids.append(centroid)
    plt.scatter(centroid[0], centroid[1], c=plt.cm.tab10(digit/9), 
               s=200, marker='o', edgecolor='black', linewidth=2, label=str(digit))
plt.scatter(custom_umap_2d[0, 0], custom_umap_2d[0, 1], 
           c='red', marker='x', s=300, linewidth=4, label='Your Digit')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('Digit Class Centroids')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 7: Silhouette comparison
plt.subplot(3, 3, 7)
methods = ['UMAP', 'PCA', 't-SNE']
scores = [silhouette_umap, silhouette_pca, silhouette_tsne]
bars = plt.bar(methods, scores, color=['blue', 'orange', 'green'], alpha=0.7)
plt.ylabel('Silhouette Score')
plt.title('Method Comparison: Digit Separation')
plt.ylim(0, max(scores) * 1.1)
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

# Plot 8: Sample digits from each class
plt.subplot(3, 3, 8)
sample_indices = []
for digit in range(10):
    digit_indices = np.where(y_digits == digit)[0]
    sample_indices.append(digit_indices[0])

for i, idx in enumerate(sample_indices):
    plt.subplot(3, 5, 11 + i)  # Start from position 11 in a 3x5 grid
    digit_img = X_digits[idx].reshape(8, 8)
    plt.imshow(digit_img, cmap='gray', interpolation='nearest')
    plt.title(f'Digit {y_digits[idx]}', fontsize=8)
    plt.axis('off')

# Plot 9: 3D UMAP visualization
if X_umap_3d.shape[1] >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(3, 3, 9, projection='3d')
    scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], 
                        c=y_digits, cmap='tab10', alpha=0.6)
    ax.scatter(custom_umap_3d[0, 0], custom_umap_3d[0, 1], custom_umap_3d[0, 2], 
              c='black', marker='x', s=200, linewidth=3)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')
    ax.set_title('UMAP 3D: Digits')

plt.tight_layout()
plt.savefig('digits_umap_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nDigits analysis complete! Plot saved as 'digits_umap_analysis.png'")