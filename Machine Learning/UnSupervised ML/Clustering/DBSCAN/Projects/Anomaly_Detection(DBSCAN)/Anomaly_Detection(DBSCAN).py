import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

# DATA LOAD
data = pd.read_csv('network_anomaly_dataset.csv')
copyData = data.copy()

print("=== Network Anomaly Detection with DBSCAN ===")
print(f"Dataset shape: {copyData.shape}")
print(copyData.head())
print(f"\nDataset info:")
print(copyData.describe())

# Check for anomaly labels if available
if 'is_anomaly' in copyData.columns:
    print(f"\nAnomaly distribution:")
    print(copyData['is_anomaly'].value_counts())

# MODEL
feature_columns = ['packet_size', 'connection_duration', 'bytes_sent', 'bytes_received', 'port_number']
X = copyData[feature_columns]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal epsilon using k-distance graph
k = 5  # MinPts = 5 for anomaly detection
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)
distances = np.sort(distances[:, k-1], axis=0)

# DBSCAN with different epsilon values for anomaly detection
eps_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
models = {}

print(f"\n=== DBSCAN Parameter Tuning for Anomaly Detection ===")
for eps in eps_values:
    model = DBSCAN(eps=eps, min_samples=k)
    clusters = model.fit_predict(X_scaled)
    
    unique_labels = set(clusters)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_anomalies = list(clusters).count(-1)
    anomaly_rate = n_anomalies / len(clusters)
    
    models[eps] = {
        'model': model,
        'clusters': clusters,
        'n_clusters': n_clusters,
        'n_anomalies': n_anomalies,
        'anomaly_rate': anomaly_rate
    }
    
    print(f"Eps: {eps}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Anomalies: {n_anomalies} ({anomaly_rate:.1%})")

# Choose optimal epsilon (typically want 1-5% anomaly rate)
optimal_eps = 0.5
best_model = models[optimal_eps]['model']
best_clusters = models[optimal_eps]['clusters']
copyData['cluster'] = best_clusters
copyData['is_anomaly_detected'] = (best_clusters == -1)

# METRICS
n_clusters = models[optimal_eps]['n_clusters']
n_anomalies = models[optimal_eps]['n_anomalies']
anomaly_rate = models[optimal_eps]['anomaly_rate']

print(f"\n=== Anomaly Detection Results (eps={optimal_eps}) ===")
print(f"Number of clusters: {n_clusters}")
print(f"Number of anomalies detected: {n_anomalies}")
print(f"Anomaly rate: {anomaly_rate:.2%}")

# Calculate detection accuracy if ground truth available
if 'is_anomaly' in copyData.columns:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(copyData['is_anomaly'], copyData['is_anomaly_detected'])
    precision = precision_score(copyData['is_anomaly'], copyData['is_anomaly_detected'])
    recall = recall_score(copyData['is_anomaly'], copyData['is_anomaly_detected'])
    f1 = f1_score(copyData['is_anomaly'], copyData['is_anomaly_detected'])
    
    print(f"\n=== Detection Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    cm = confusion_matrix(copyData['is_anomaly'], copyData['is_anomaly_detected'])
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Analyze anomalies
print(f"\n=== Anomaly Analysis ===")
normal_data = copyData[copyData['cluster'] != -1]
anomaly_data = copyData[copyData['cluster'] == -1]

if len(anomaly_data) > 0:
    print(f"Normal data statistics:")
    print(normal_data[feature_columns].describe())
    print(f"\nAnomaly data statistics:")
    print(anomaly_data[feature_columns].describe())

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: K-distance graph
plt.subplot(3, 3, 1)
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-NN Distance')
plt.title('K-Distance Graph for Epsilon Selection')
plt.grid(True)

# Plot 2: Anomaly rate vs epsilon
plt.subplot(3, 3, 2)
eps_list = list(models.keys())
anomaly_rates = [models[eps]['anomaly_rate'] for eps in eps_list]
plt.plot(eps_list, anomaly_rates, 'ro-', linewidth=2)
plt.xlabel('Epsilon')
plt.ylabel('Anomaly Rate')
plt.title('Anomaly Rate vs Epsilon')
plt.grid(True)

# Plot 3: Main anomaly detection result
plt.subplot(3, 3, 3)
normal_mask = best_clusters != -1
anomaly_mask = best_clusters == -1

# Plot normal points
plt.scatter(copyData.loc[normal_mask, 'packet_size'], 
           copyData.loc[normal_mask, 'connection_duration'], 
           c=best_clusters[normal_mask], cmap='viridis', alpha=0.7, s=50, label='Normal')

# Plot anomalies
if np.any(anomaly_mask):
    plt.scatter(copyData.loc[anomaly_mask, 'packet_size'], 
               copyData.loc[anomaly_mask, 'connection_duration'], 
               c='red', marker='x', s=100, alpha=0.8, label='Anomaly')

plt.xlabel('Packet Size')
plt.ylabel('Connection Duration')
plt.title(f'Anomaly Detection (eps={optimal_eps})')
plt.legend()

# Plot 4: Confusion matrix if ground truth available
if 'is_anomaly' in copyData.columns:
    plt.subplot(3, 3, 4)
    cm = confusion_matrix(copyData['is_anomaly'], copyData['is_anomaly_detected'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

# Plot 5: Feature distribution comparison
plt.subplot(3, 3, 5)
feature_to_plot = 'bytes_sent'
if len(anomaly_data) > 0:
    plt.hist(normal_data[feature_to_plot], bins=30, alpha=0.7, label='Normal', color='blue', density=True)
    plt.hist(anomaly_data[feature_to_plot], bins=30, alpha=0.7, label='Anomaly', color='red', density=True)
    plt.xlabel(feature_to_plot)
    plt.ylabel('Density')
    plt.title(f'{feature_to_plot} Distribution')
    plt.legend()

# Plot 6: Anomaly scores (distance to nearest cluster)
plt.subplot(3, 3, 6)
# Calculate distance to nearest normal point for each anomaly
if len(anomaly_data) > 0:
    from scipy.spatial.distance import cdist
    
    normal_points = X_scaled[normal_mask]
    anomaly_points = X_scaled[anomaly_mask]
    
    if len(normal_points) > 0:
        distances_to_normal = cdist(anomaly_points, normal_points, metric='euclidean')
        min_distances = np.min(distances_to_normal, axis=1)
        
        plt.hist(min_distances, bins=20, alpha=0.7, color='orange')
        plt.xlabel('Distance to Nearest Normal Point')
        plt.ylabel('Frequency')
        plt.title('Anomaly Scores Distribution')

# CUSTOM INPUT
print("\n=== Custom Network Traffic Analysis ===")
print("Enter network traffic characteristics:")
packet_size = float(input("Enter packet size (bytes, 0-10000): "))
duration = float(input("Enter connection duration (seconds, 0-300): "))
bytes_sent = float(input("Enter bytes sent (0-1000000): "))
bytes_received = float(input("Enter bytes received (0-1000000): "))
port_number = float(input("Enter port number (0-65535): "))

custom_data = np.array([[packet_size, duration, bytes_sent, bytes_received, port_number]])
custom_data_scaled = scaler.transform(custom_data)

# Check if custom point is anomaly
# Find distance to nearest core point
core_points = X_scaled[best_model.core_sample_indices_] if len(best_model.core_sample_indices_) > 0 else X_scaled
if len(core_points) > 0:
    distances_to_core = np.sqrt(np.sum((core_points - custom_data_scaled)**2, axis=1))
    min_distance = np.min(distances_to_core)
    
    is_anomaly = min_distance > optimal_eps
    
    print(f"\nCustom Traffic Analysis:")
    print(f"Packet Size: {packet_size}")
    print(f"Duration: {duration}")
    print(f"Bytes Sent: {bytes_sent}")
    print(f"Bytes Received: {bytes_received}")
    print(f"Port Number: {port_number}")
    print(f"Distance to nearest normal traffic: {min_distance:.4f}")
    print(f"Classification: {'ANOMALY' if is_anomaly else 'NORMAL'}")
    
    if is_anomaly:
        print("⚠️  This traffic pattern appears suspicious!")
    else:
        print("✅ This traffic pattern appears normal.")

# Plot 7: Custom point visualization
plt.subplot(3, 3, 7)
plt.scatter(copyData.loc[normal_mask, 'packet_size'], 
           copyData.loc[normal_mask, 'bytes_sent'], 
           c=best_clusters[normal_mask], cmap='viridis', alpha=0.7, s=50, label='Normal')

if np.any(anomaly_mask):
    plt.scatter(copyData.loc[anomaly_mask, 'packet_size'], 
               copyData.loc[anomaly_mask, 'bytes_sent'], 
               c='red', marker='x', s=100, alpha=0.8, label='Detected Anomaly')

plt.scatter(packet_size, bytes_sent, c='black', marker='X', s=300, 
           label='Your Traffic', edgecolors='white', linewidths=2)
plt.xlabel('Packet Size')
plt.ylabel('Bytes Sent')
plt.title('Your Traffic Analysis')
plt.legend()

# Plot 8: Feature correlation for anomalies
plt.subplot(3, 3, 8)
if len(anomaly_data) > 3:
    correlation_matrix = anomaly_data[feature_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Anomaly Feature Correlations')
else:
    plt.text(0.5, 0.5, 'Insufficient\nanomalies for\ncorrelation analysis', 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12)
    plt.title('Anomaly Feature Correlations')

# Plot 9: Detection performance over different epsilon values
plt.subplot(3, 3, 9)
if 'is_anomaly' in copyData.columns:
    eps_list = list(models.keys())
    precisions = []
    recalls = []
    
    for eps in eps_list:
        clusters = models[eps]['clusters']
        detected_anomalies = (clusters == -1)
        
        precision = precision_score(copyData['is_anomaly'], detected_anomalies)
        recall = recall_score(copyData['is_anomaly'], detected_anomalies)
        
        precisions.append(precision)
        recalls.append(recall)
    
    plt.plot(eps_list, precisions, 'b-o', label='Precision')
    plt.plot(eps_list, recalls, 'r-s', label='Recall')
    plt.xlabel('Epsilon')
    plt.ylabel('Score')
    plt.title('Precision vs Recall')
    plt.legend()
    plt.grid(True)
else:
    # Alternative plot if no ground truth
    eps_list = list(models.keys())
    n_clusters_list = [models[eps]['n_clusters'] for eps in eps_list]
    plt.plot(eps_list, n_clusters_list, 'g-o')
    plt.xlabel('Epsilon')
    plt.ylabel('Number of Clusters')
    plt.title('Clusters vs Epsilon')
    plt.grid(True)

plt.tight_layout()
plt.savefig('network_anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'network_anomaly_detection.png'")