import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import seaborn as sns

# DATA LOAD
np.random.seed(42)

# Generate synthetic customer data
n_customers = 800

# Generate base features using make_classification
X_base, y_segments = make_classification(
    n_samples=n_customers,
    n_features=15,
    n_informative=10,
    n_redundant=3,
    n_clusters_per_class=2,
    class_sep=1.2,
    random_state=42
)

# Create realistic customer features
customer_data = {
    'age': np.random.normal(40, 12, n_customers).clip(18, 80),
    'annual_income': np.random.lognormal(10.5, 0.5, n_customers).clip(20000, 200000),
    'spending_score': np.random.normal(50, 20, n_customers).clip(1, 100),
    'family_size': np.random.choice([1, 2, 3, 4, 5, 6], n_customers, p=[0.2, 0.3, 0.25, 0.15, 0.08, 0.02]),
    'education_years': np.random.normal(14, 3, n_customers).clip(8, 20),
    'work_experience': np.random.normal(15, 8, n_customers).clip(0, 45),
    'credit_score': np.random.normal(650, 100, n_customers).clip(300, 850),
    'num_purchases': np.random.poisson(12, n_customers),
    'avg_purchase_amount': np.random.lognormal(5, 0.8, n_customers).clip(10, 5000),
    'days_since_last_purchase': np.random.exponential(30, n_customers).clip(0, 365),
    'num_returns': np.random.poisson(2, n_customers),
    'loyalty_points': np.random.exponential(1000, n_customers).clip(0, 10000),
    'website_visits': np.random.poisson(8, n_customers),
    'email_opens': np.random.poisson(5, n_customers),
    'social_media_followers': np.random.exponential(500, n_customers).clip(0, 50000)
}

# Add the base classification features
for i in range(X_base.shape[1]):
    customer_data[f'feature_{i+1}'] = X_base[:, i]

# Create DataFrame
df = pd.DataFrame(customer_data)
df['customer_segment'] = y_segments  # True customer segments
copyData = df.copy()

feature_names = [col for col in df.columns if col != 'customer_segment']

print("=== UMAP for Customer Data Analysis ===")
print(f"Dataset shape: {copyData.shape}")
print(f"Customer features: {len(feature_names)}")
print(f"Customer segments: {len(np.unique(y_segments))}")
print(f"Samples per segment: {np.bincount(y_segments)}")
print(copyData.head())

# MODEL
X = copyData[feature_names]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n=== UMAP Customer Analysis ===")

# UMAP with optimized parameters for customer data
umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.1, metric='euclidean')
umap_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=20, min_dist=0.1, metric='euclidean')

X_umap_2d = umap_2d.fit_transform(X_scaled)
X_umap_3d = umap_3d.fit_transform(X_scaled)

print(f"Original dimensions: {X_scaled.shape[1]} customer features")
print(f"UMAP 2D shape: {X_umap_2d.shape}")
print(f"UMAP 3D shape: {X_umap_3d.shape}")

# Compare with PCA and t-SNE
pca_2d = PCA(n_components=2, random_state=42)
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)

X_pca_2d = pca_2d.fit_transform(X_scaled)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)

# Perform K-means clustering in UMAP space
kmeans = KMeans(n_clusters=3, random_state=42)
umap_clusters = kmeans.fit_predict(X_umap_2d)

print(f"\n=== Method Comparison for Customer Data ===")
print(f"PCA explained variance: {sum(pca_2d.explained_variance_ratio_):.4f}")
print(f"UMAP reveals customer behavior patterns and segments")
print(f"K-means found {len(np.unique(umap_clusters))} clusters in UMAP space")

# CUSTOM INPUT
print("\n=== Custom Customer Profile ===")
print("Enter your customer profile to see where you fit in the customer space:")

custom_customer = {}
feature_descriptions = {
    'age': 'Age (18-80)',
    'annual_income': 'Annual Income ($20,000-$200,000)',
    'spending_score': 'Spending Score (1-100)',
    'family_size': 'Family Size (1-6)',
    'education_years': 'Education Years (8-20)',
    'work_experience': 'Work Experience Years (0-45)',
    'credit_score': 'Credit Score (300-850)',
    'num_purchases': 'Number of Purchases per Year',
    'avg_purchase_amount': 'Average Purchase Amount ($)',
    'days_since_last_purchase': 'Days Since Last Purchase (0-365)',
    'num_returns': 'Number of Returns per Year',
    'loyalty_points': 'Loyalty Points (0-10,000)',
    'website_visits': 'Website Visits per Month',
    'email_opens': 'Email Opens per Month',
    'social_media_followers': 'Social Media Followers'
}

try:
    print("\nEnter values for key customer features:")
    for feature, description in feature_descriptions.items():
        while True:
            try:
                value = input(f"{description}: ").strip()
                if value.lower() in ['skip', 'default', '']:
                    # Use mean value
                    custom_customer[feature] = copyData[feature].mean()
                    break
                custom_customer[feature] = float(value)
                break
            except ValueError:
                print("Please enter a valid number or 'skip' for default")
    
    # Use mean values for the additional features
    for feature in feature_names:
        if feature not in custom_customer:
            custom_customer[feature] = copyData[feature].mean()
    
    # Create customer array
    custom_array = np.array([custom_customer[feature] for feature in feature_names]).reshape(1, -1)
    custom_scaled = scaler.transform(custom_array)
    custom_umap_2d = umap_2d.transform(custom_scaled)
    custom_umap_3d = umap_3d.transform(custom_scaled)
    
    print(f"\nYour position in customer space: ({custom_umap_2d[0, 0]:.3f}, {custom_umap_2d[0, 1]:.3f})")
    
    # Find nearest customers
    nn_model = NearestNeighbors(n_neighbors=5)
    nn_model.fit(X_umap_2d)
    distances, indices = nn_model.kneighbors(custom_umap_2d)
    
    print(f"\n=== Similar Customers ===")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        similar_customer = copyData.iloc[idx]
        print(f"{i+1}. Customer {idx}: distance={dist:.3f}")
        print(f"   Age: {similar_customer['age']:.0f}, Income: ${similar_customer['annual_income']:.0f}")
        print(f"   Spending Score: {similar_customer['spending_score']:.0f}, Segment: {similar_customer['customer_segment']}")
    
    # Predict customer segment
    predicted_cluster = kmeans.predict(custom_umap_2d)[0]
    predicted_segment = y_segments[indices[0][0]]  # Use nearest neighbor's segment
    print(f"\nPredicted customer segment: {predicted_segment}")
    print(f"UMAP cluster assignment: {predicted_cluster}")

except KeyboardInterrupt:
    print("\nUsing average customer profile for visualization...")
    custom_array = X_scaled.mean(axis=0).reshape(1, -1)
    custom_umap_2d = umap_2d.transform(custom_array)
    custom_umap_3d = umap_3d.transform(custom_array)
    
    nn_model = NearestNeighbors(n_neighbors=5)
    nn_model.fit(X_umap_2d)
    distances, indices = nn_model.kneighbors(custom_umap_2d)
    predicted_cluster = kmeans.predict(custom_umap_2d)[0]

# METRICS
print(f"\n=== Customer Segmentation Metrics ===")

# Calculate silhouette scores
from sklearn.metrics import silhouette_score, adjusted_rand_score

silhouette_umap_true = silhouette_score(X_umap_2d, y_segments)
silhouette_umap_kmeans = silhouette_score(X_umap_2d, umap_clusters)
silhouette_pca = silhouette_score(X_pca_2d, y_segments)
silhouette_tsne = silhouette_score(X_tsne_2d, y_segments)

ari_score = adjusted_rand_score(y_segments, umap_clusters)

print(f"Silhouette scores (true segments):")
print(f"  UMAP: {silhouette_umap_true:.4f}")
print(f"  PCA:  {silhouette_pca:.4f}")
print(f"  t-SNE: {silhouette_tsne:.4f}")
print(f"Silhouette score (K-means on UMAP): {silhouette_umap_kmeans:.4f}")
print(f"Adjusted Rand Index (true vs UMAP clusters): {ari_score:.4f}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: UMAP customer segments (true labels)
plt.subplot(3, 3, 1)
scatter = plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], c=y_segments, cmap='viridis', alpha=0.7)
plt.scatter(custom_umap_2d[0, 0], custom_umap_2d[0, 1], 
           c='black', marker='x', s=200, linewidth=3, label='Your Profile')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP: True Customer Segments')
plt.legend()
plt.colorbar(scatter)

# Plot 2: UMAP K-means clusters
plt.subplot(3, 3, 2)
scatter = plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], c=umap_clusters, cmap='tab10', alpha=0.7)
plt.scatter(custom_umap_2d[0, 0], custom_umap_2d[0, 1], 
           c='black', marker='x', s=200, linewidth=3, label='Your Profile')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP: K-means Clusters')
plt.legend()
plt.colorbar(scatter)

# Plot 3: PCA comparison
plt.subplot(3, 3, 3)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_segments, cmap='viridis', alpha=0.7)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA: Customer Segments')
plt.colorbar()

# Plot 4: Income vs Spending colored by UMAP clusters
plt.subplot(3, 3, 4)
plt.scatter(copyData['annual_income'], copyData['spending_score'], 
           c=umap_clusters, cmap='tab10', alpha=0.7)
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score')
plt.title('Income vs Spending (UMAP Clusters)')
plt.colorbar()

# Plot 5: Age vs Credit Score colored by segments
plt.subplot(3, 3, 5)
plt.scatter(copyData['age'], copyData['credit_score'], 
           c=y_segments, cmap='viridis', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Credit Score')
plt.title('Age vs Credit Score (True Segments)')
plt.colorbar()

# Plot 6: Nearest customers
plt.subplot(3, 3, 6)
plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], c='lightgray', alpha=0.3)
colors = ['red', 'orange', 'yellow', 'green', 'blue']
for i, idx in enumerate(indices[0]):
    plt.scatter(X_umap_2d[idx, 0], X_umap_2d[idx, 1], 
               c=colors[i], s=100, alpha=0.8, 
               label=f'Customer {idx} (Seg {y_segments[idx]})')
plt.scatter(custom_umap_2d[0, 0], custom_umap_2d[0, 1], 
           c='black', marker='x', s=200, linewidth=3, label='Your Profile')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('Similar Customers')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 7: Feature importance (correlation with UMAP components)
plt.subplot(3, 3, 7)
correlations = []
important_features = ['age', 'annual_income', 'spending_score', 'credit_score', 
                     'num_purchases', 'avg_purchase_amount', 'loyalty_points']
for feature in important_features:
    corr = np.corrcoef(copyData[feature], X_umap_2d[:, 0])[0, 1]
    correlations.append(abs(corr))

plt.barh(range(len(important_features)), correlations, color='skyblue', alpha=0.7)
plt.yticks(range(len(important_features)), important_features)
plt.xlabel('|Correlation| with UMAP1')
plt.title('Feature Importance in UMAP')

# Plot 8: Segment characteristics
plt.subplot(3, 3, 8)
segment_means = copyData.groupby('customer_segment')[['annual_income', 'spending_score']].mean()
segments = segment_means.index
x_pos = np.arange(len(segments))
width = 0.35

plt.bar(x_pos - width/2, segment_means['annual_income']/1000, width, 
        label='Income (K$)', alpha=0.7)
plt.bar(x_pos + width/2, segment_means['spending_score'], width, 
        label='Spending Score', alpha=0.7)
plt.xlabel('Customer Segment')
plt.ylabel('Value')
plt.title('Segment Characteristics')
plt.xticks(x_pos, segments)
plt.legend()

# Plot 9: 3D UMAP visualization
if X_umap_3d.shape[1] >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(3, 3, 9, projection='3d')
    scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], 
                        c=y_segments, cmap='viridis', alpha=0.6)
    ax.scatter(custom_umap_3d[0, 0], custom_umap_3d[0, 1], custom_umap_3d[0, 2], 
              c='black', marker='x', s=200, linewidth=3)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')
    ax.set_title('UMAP 3D: Customer Space')

plt.tight_layout()
plt.savefig('customer_umap_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nCustomer analysis complete! Plot saved as 'customer_umap_analysis.png'")