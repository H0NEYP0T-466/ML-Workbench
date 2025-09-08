import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# DATA LOAD
np.random.seed(42)

# Generate multi-class classification dataset
X_high_dim, y_labels = make_classification(
    n_samples=600, 
    n_features=20, 
    n_informative=15, 
    n_redundant=3,
    n_classes=4,
    n_clusters_per_class=2,
    class_sep=1.2,
    random_state=42
)

# Create DataFrame
feature_names = [f'feature_{i+1}' for i in range(20)]
df = pd.DataFrame(X_high_dim, columns=feature_names)
df['target'] = y_labels
copyData = df.copy()

print("=== Linear Discriminant Analysis (LDA) ===")
print(f"Dataset shape: {copyData.shape}")
print(copyData.head())
print(f"\nDataset info:")
print(f"Features: {len(feature_names)}")
print(f"Classes: {len(np.unique(y_labels))}")
print(f"Samples per class: {np.bincount(y_labels)}")

# MODEL
X = copyData[feature_names]
y = copyData['target']

# Split data for supervised learning aspect
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)

# LDA for dimensionality reduction and classification
# Maximum components is min(n_features, n_classes-1)
max_components = min(X_scaled.shape[1], len(np.unique(y)) - 1)
print(f"\nMaximum LDA components: {max_components}")

# Apply LDA with different numbers of components
lda_results = {}
for n_comp in range(1, max_components + 1):
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    X_lda_train = lda.fit_transform(X_train_scaled, y_train)
    X_lda_test = lda.transform(X_test_scaled)
    
    # Classification performance
    lda_classifier = LinearDiscriminantAnalysis()
    lda_classifier.fit(X_lda_train, y_train)
    y_pred = lda_classifier.predict(X_lda_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    lda_results[n_comp] = {
        'model': lda,
        'classifier': lda_classifier,
        'X_train_lda': X_lda_train,
        'X_test_lda': X_lda_test,
        'accuracy': accuracy,
        'explained_variance_ratio': lda.explained_variance_ratio_
    }

# Choose optimal number of components (balance between dimensionality reduction and accuracy)
optimal_components = 2  # Good for visualization
lda_model = lda_results[optimal_components]['model']
lda_classifier = lda_results[optimal_components]['classifier']

# Apply LDA to full dataset for visualization
X_lda_full = lda_model.fit_transform(X_scaled, y)

# Create DataFrame with LDA results
lda_columns = [f'LD{i+1}' for i in range(optimal_components)]
df_lda = pd.DataFrame(X_lda_full, columns=lda_columns)
df_lda['target'] = y

# METRICS
print(f"\n=== LDA Metrics ===")
print(f"Optimal components: {optimal_components}")
print(f"Explained variance ratio: {lda_model.explained_variance_ratio_}")
print(f"Total variance explained: {np.sum(lda_model.explained_variance_ratio_):.4f}")

# Classification performance
y_pred_full = lda_classifier.predict(lda_model.transform(X_scaled))
accuracy_full = accuracy_score(y, y_pred_full)
print(f"Classification accuracy: {accuracy_full:.4f}")

# Component analysis
print(f"\n=== Linear Discriminant Components ===")
for i in range(optimal_components):
    print(f"\nLD{i+1} (explains {lda_model.explained_variance_ratio_[i]*100:.2f}% variance):")
    # Get feature weights for this component
    component_weights = lda_model.scalings_[:, i]
    sorted_features = sorted(zip(feature_names, component_weights), key=lambda x: abs(x[1]), reverse=True)
    for feature, weight in sorted_features[:3]:
        print(f"  {feature}: {weight:.4f}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: Explained variance by component
plt.subplot(3, 3, 1)
components = range(1, max_components + 1)
variances = [np.sum(lda_results[c]['explained_variance_ratio']) for c in components]
plt.plot(components, variances, 'bo-', linewidth=2)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('LDA Explained Variance')
plt.grid(True)

# Plot 2: Classification accuracy vs components
plt.subplot(3, 3, 2)
accuracies = [lda_results[c]['accuracy'] for c in components]
plt.plot(components, accuracies, 'ro-', linewidth=2)
plt.xlabel('Number of Components')
plt.ylabel('Classification Accuracy')
plt.title('Accuracy vs Number of Components')
plt.grid(True)

# Plot 3: LDA 2D projection
plt.subplot(3, 3, 3)
scatter = plt.scatter(df_lda['LD1'], df_lda['LD2'], c=df_lda['target'], cmap='viridis', alpha=0.7)
plt.xlabel(f'LD1 ({lda_model.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'LD2 ({lda_model.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('LDA 2D Projection')
plt.colorbar(scatter)

# Plot 4: Class separation analysis
plt.subplot(3, 3, 4)
unique_classes = np.unique(y)
colors = ['blue', 'orange', 'green', 'red']
for i, class_label in enumerate(unique_classes):
    mask = y == class_label
    plt.scatter(X_lda_full[mask, 0], X_lda_full[mask, 1], 
               c=colors[i % len(colors)], label=f'Class {class_label}', alpha=0.7)

plt.xlabel(f'LD1 ({lda_model.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'LD2 ({lda_model.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Class Separation in LDA Space')
plt.legend()

# Plot 5: Feature loadings heatmap
plt.subplot(3, 3, 5)
loadings = lda_model.scalings_[:, :min(3, optimal_components)]
loadings_df = pd.DataFrame(loadings, index=feature_names, columns=lda_columns[:min(3, optimal_components)])
sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Feature Loadings in LDA Space')
plt.ylabel('Original Features')

# Plot 6: Decision boundaries (for 2D case)
if optimal_components == 2:
    plt.subplot(3, 3, 6)
    h = 0.1
    x_min, x_max = X_lda_full[:, 0].min() - 1, X_lda_full[:, 0].max() + 1
    y_min, y_max = X_lda_full[:, 1].min() - 1, X_lda_full[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = lda_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = plt.scatter(X_lda_full[:, 0], X_lda_full[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title('LDA Decision Boundaries')
    plt.colorbar(scatter)

# CUSTOM INPUT
print("\n=== Custom Data Point Classification ===")
print("Enter values for first 5 features:")
custom_input = []
for i in range(5):
    value = float(input(f"Enter feature_{i+1} value (-3 to 3): "))
    custom_input.append(value)

# Pad with zeros for remaining features
custom_input.extend([0] * (len(feature_names) - 5))
custom_data = np.array([custom_input])
custom_data_scaled = scaler.transform(custom_data)
custom_lda = lda_model.transform(custom_data_scaled)
custom_prediction = lda_classifier.predict(custom_lda)
custom_proba = lda_classifier.predict_proba(custom_lda)

print(f"\nCustom Point Analysis:")
print(f"Original features (first 5): {custom_input[:5]}")
print(f"LDA transformed: {custom_lda[0]}")
print(f"Predicted class: {custom_prediction[0]}")
print(f"Class probabilities: {custom_proba[0]}")

# Plot 7: Custom point visualization
plt.subplot(3, 3, 7)
scatter = plt.scatter(df_lda['LD1'], df_lda['LD2'], c=df_lda['target'], cmap='viridis', alpha=0.7)
plt.scatter(custom_lda[0, 0], custom_lda[0, 1], c='red', marker='X', s=300, label='Your Point', edgecolors='white', linewidths=2)
plt.xlabel(f'LD1 ({lda_model.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'LD2 ({lda_model.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Your Point in LDA Space')
plt.legend()
plt.colorbar(scatter)

# Plot 8: Component importance
plt.subplot(3, 3, 8)
component_importance = np.abs(lda_model.scalings_).sum(axis=1)
sorted_features = sorted(zip(feature_names, component_importance), key=lambda x: x[1], reverse=True)
features, importances = zip(*sorted_features[:10])

plt.barh(range(len(features)), importances, color='purple', alpha=0.7)
plt.yticks(range(len(features)), features)
plt.xlabel('Total Absolute Loading')
plt.title('Feature Importance in LDA')
plt.gca().invert_yaxis()

# Plot 9: Between-class vs Within-class scatter
plt.subplot(3, 3, 9)
# Calculate class means in original space
class_means = []
for class_label in unique_classes:
    mask = y == class_label
    class_mean = X_scaled[mask].mean(axis=0)
    class_means.append(class_mean)

class_means = np.array(class_means)

# Project class means to LDA space
class_means_lda = lda_model.transform(class_means)

# Plot class centroids
for i, class_label in enumerate(unique_classes):
    plt.scatter(class_means_lda[i, 0], class_means_lda[i, 1], 
               c=colors[i % len(colors)], marker='s', s=200, 
               label=f'Class {class_label} Center', edgecolors='black', linewidths=2)

# Plot data points with reduced alpha
scatter = plt.scatter(df_lda['LD1'], df_lda['LD2'], c=df_lda['target'], cmap='viridis', alpha=0.3, s=20)
plt.scatter(custom_lda[0, 0], custom_lda[0, 1], c='red', marker='X', s=300, label='Your Point', edgecolors='white', linewidths=2)
plt.xlabel(f'LD1 ({lda_model.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'LD2 ({lda_model.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Class Centers in LDA Space')
plt.legend()

plt.tight_layout()
plt.savefig('lda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'lda_analysis.png'")