import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# Load data
data = pd.read_csv("banknote_dataset.csv")

copyData = data.copy()

# Features and target
X = copyData[["variance", "skewness", "curtosis", "entropy"]]
copyData["authentic"] = copyData["authentic"].map({"No": 0, "Yes": 1})
y = copyData["authentic"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print metrics
print("=== Banknote Authentication Results ===")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter banknote image features for authentication:")
variance = float(input("Enter variance of wavelet transformed image (-5 to 8): "))
skewness = float(input("Enter skewness of wavelet transformed image (-8 to 8): "))
curtosis = float(input("Enter curtosis of wavelet transformed image (-8 to 8): "))
entropy = float(input("Enter entropy of image (-5 to 5): "))

custom_data = pd.DataFrame({
    "variance": [variance],
    "skewness": [skewness],
    "curtosis": [curtosis],
    "entropy": [entropy]
})

custom_prediction = model.predict(custom_data)
prediction_proba = model.predict_proba(custom_data)

if custom_prediction == 1:
    print(f"This banknote is AUTHENTIC ✅ (Probability: {prediction_proba[0][1]:.3f})")
else:
    print(f"This banknote is FAKE ⚠️ (Probability: {prediction_proba[0][0]:.3f})")

# PLOTS

# 1. Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Authentic"], yticklabels=["Fake", "Authentic"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Banknote Authentication")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Correct vs Wrong Predictions (using first two features)
correct_idx = predictions == y_test
wrong_idx = predictions != y_test

plt.figure(figsize=(8,6))

# Correct predictions
plt.scatter(X_test.loc[correct_idx, "variance"], 
            X_test.loc[correct_idx, "skewness"], 
            color="green", alpha=0.6, label=f"Correct ✅ ({correct_idx.sum()})")

# Wrong predictions
plt.scatter(X_test.loc[wrong_idx, "variance"], 
            X_test.loc[wrong_idx, "skewness"], 
            color="red", alpha=0.6, label=f"Wrong ❌ ({wrong_idx.sum()})")

# Custom input
plt.scatter(variance, skewness, color="black", marker="X", s=200, label="Your Input ⭐")

plt.xlabel("Variance")
plt.ylabel("Skewness")
plt.title("Banknote Authentication: Correct vs Wrong Predictions")
plt.legend()
plt.savefig("correct_vs_wrong.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Feature Distribution Analysis
feature_names = ["variance", "skewness", "curtosis", "entropy"]

plt.figure(figsize=(12, 8))

for i, feature in enumerate(feature_names, 1):
    plt.subplot(2, 2, i)
    
    # Split by authenticity
    authentic_data = copyData[copyData["authentic"] == 1][feature]
    fake_data = copyData[copyData["authentic"] == 0][feature]
    
    plt.hist(authentic_data, alpha=0.7, label="Authentic", bins=20, color='green', density=True)
    plt.hist(fake_data, alpha=0.7, label="Fake", bins=20, color='red', density=True)
    
    # Mark user input
    if feature == "variance":
        plt.axvline(variance, color='black', linestyle='--', linewidth=2, label='Your Input')
    elif feature == "skewness":
        plt.axvline(skewness, color='black', linestyle='--', linewidth=2, label='Your Input')
    elif feature == "curtosis":
        plt.axvline(curtosis, color='black', linestyle='--', linewidth=2, label='Your Input')
    elif feature == "entropy":
        plt.axvline(entropy, color='black', linestyle='--', linewidth=2, label='Your Input')
    
    plt.xlabel(feature.title())
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"{feature.title()} Distribution")

plt.suptitle("Feature Distributions: Authentic vs Fake Banknotes")
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Pairwise Feature Relationships
plt.figure(figsize=(12, 10))

# Create pairwise scatter plots for the most informative feature pairs
feature_pairs = [("variance", "skewness"), ("variance", "curtosis"), 
                ("variance", "entropy"), ("skewness", "curtosis"),
                ("skewness", "entropy"), ("curtosis", "entropy")]

for i, (feat1, feat2) in enumerate(feature_pairs, 1):
    plt.subplot(2, 3, i)
    
    # Plot authentic and fake banknotes
    authentic_data = copyData[copyData["authentic"] == 1]
    fake_data = copyData[copyData["authentic"] == 0]
    
    plt.scatter(authentic_data[feat1], authentic_data[feat2], 
               alpha=0.6, label="Authentic", color='green', s=30)
    plt.scatter(fake_data[feat1], fake_data[feat2], 
               alpha=0.6, label="Fake", color='red', s=30)
    
    # Plot user input if it matches the current feature pair
    if feat1 == "variance" and feat2 == "skewness":
        plt.scatter(variance, skewness, color='black', marker='X', s=100, label='Your Input')
    elif feat1 == "variance" and feat2 == "curtosis":
        plt.scatter(variance, curtosis, color='black', marker='X', s=100, label='Your Input')
    elif feat1 == "variance" and feat2 == "entropy":
        plt.scatter(variance, entropy, color='black', marker='X', s=100, label='Your Input')
    elif feat1 == "skewness" and feat2 == "curtosis":
        plt.scatter(skewness, curtosis, color='black', marker='X', s=100, label='Your Input')
    elif feat1 == "skewness" and feat2 == "entropy":
        plt.scatter(skewness, entropy, color='black', marker='X', s=100, label='Your Input')
    elif feat1 == "curtosis" and feat2 == "entropy":
        plt.scatter(curtosis, entropy, color='black', marker='X', s=100, label='Your Input')
    
    plt.xlabel(feat1.title())
    plt.ylabel(feat2.title())
    plt.title(f"{feat1.title()} vs {feat2.title()}")
    if i == 1:  # Only show legend for first subplot
        plt.legend()

plt.suptitle("Pairwise Feature Relationships")
plt.tight_layout()
plt.savefig("pairwise_features.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. Decision Boundary Visualization (2D projection)
# Use the two most discriminative features
plt.figure(figsize=(10, 8))

# Create a mesh for decision boundary
h = 0.1
feat1, feat2 = "variance", "skewness"
x_min, x_max = copyData[feat1].min() - 1, copyData[feat1].max() + 1
y_min, y_max = copyData[feat2].min() - 1, copyData[feat2].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# For decision boundary, we need to provide all 4 features
# We'll use mean values for the other two features
mean_curtosis = copyData["curtosis"].mean()
mean_entropy = copyData["entropy"].mean()

mesh_features = np.c_[xx.ravel(), yy.ravel(), 
                     np.full(xx.ravel().shape, mean_curtosis),
                     np.full(xx.ravel().shape, mean_entropy)]

Z = model.predict_proba(mesh_features)[:, 1]  # Probability of authentic
Z = Z.reshape(xx.shape)

# Plot decision boundary
contour = plt.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap='RdYlGn')
plt.colorbar(contour, label='Probability of Authentic')

# Plot decision boundary line
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

# Plot data points
authentic_data = copyData[copyData["authentic"] == 1]
fake_data = copyData[copyData["authentic"] == 0]

plt.scatter(authentic_data[feat1], authentic_data[feat2], 
           alpha=0.8, label="Authentic", color='green', s=50, edgecolor='black')
plt.scatter(fake_data[feat1], fake_data[feat2], 
           alpha=0.8, label="Fake", color='red', s=50, edgecolor='black')

# Plot user input
plt.scatter(variance, skewness, color='black', marker='X', s=200, label='Your Input', edgecolor='white')

plt.xlabel(feat1.title())
plt.ylabel(feat2.title())
plt.title(f"Decision Boundary: {feat1.title()} vs {feat2.title()}\n(Other features fixed at mean values)")
plt.legend()
plt.savefig("decision_boundary.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nPlots saved as PNG files in the current directory!")