import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import random

# Generate synthetic dataset
random.seed(42)
np.random.seed(42)

data = {
    "feature1": [],
    "feature2": [],
    "class": []
}

# Generate 500+ data points with realistic patterns
for i in range(600):
    # Create overlapping distributions for Naive Bayes to work well
    if i < 300:
        # Class 0 - normal distribution centered at (3, 3)
        feature1 = np.random.normal(3, 2)
        feature2 = np.random.normal(3, 2)
        class_label = 0
    else:
        # Class 1 - normal distribution centered at (7, 7)
        feature1 = np.random.normal(7, 2)
        feature2 = np.random.normal(7, 2)
        class_label = 1
    
    data["feature1"].append(feature1)
    data["feature2"].append(feature2)
    data["class"].append(class_label)

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["feature1", "feature2"]]
y = df["class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print metrics
print("=== Naive Bayes Classification Results ===")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter values for prediction:")
feature1_input = float(input("Enter feature1 value (typically 0-12): "))
feature2_input = float(input("Enter feature2 value (typically 0-12): "))

custom_data = pd.DataFrame({
    "feature1": [feature1_input],
    "feature2": [feature2_input]
})

custom_prediction = model.predict(custom_data)
prediction_proba = model.predict_proba(custom_data)
print(f"Predicted class: {custom_prediction[0]}")
print(f"Prediction probabilities: Class 0: {prediction_proba[0][0]:.3f}, Class 1: {prediction_proba[0][1]:.3f}")

# PLOTS

# 1. Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes Classification")
plt.show()

# 2. Correct vs Wrong Predictions Scatter Plot
correct_idx = predictions == y_test
wrong_idx = predictions != y_test

plt.figure(figsize=(8,6))

# Correct predictions
plt.scatter(X_test.loc[correct_idx, "feature1"], 
            X_test.loc[correct_idx, "feature2"], 
            color="green", alpha=0.6, label=f"Correct ✅ ({correct_idx.sum()})")

# Wrong predictions
plt.scatter(X_test.loc[wrong_idx, "feature1"], 
            X_test.loc[wrong_idx, "feature2"], 
            color="red", alpha=0.6, label=f"Wrong ❌ ({wrong_idx.sum()})")

# Custom input
plt.scatter(feature1_input, feature2_input, color="black", marker="X", s=200, label="Your Input ⭐")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Naive Bayes Classification: Correct vs Wrong Predictions")
plt.legend()
plt.show()

# 3. Decision Boundary with probability contours
h = 0.1
x_min, x_max = X["feature1"].min() - 2, X["feature1"].max() + 2
y_min, y_max = X["feature2"].min() - 2, X["feature2"].max() + 2

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Get probability predictions for the mesh
mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z_proba = model.predict_proba(mesh_points)[:, 1]  # Probability of class 1
Z_proba = Z_proba.reshape(xx.shape)

plt.figure(figsize=(10, 8))

# Plot probability contours
contour = plt.contourf(xx, yy, Z_proba, levels=20, alpha=0.6, cmap='RdYlBu')
plt.colorbar(contour, label='Probability of Class 1')

# Plot decision boundary (0.5 probability)
plt.contour(xx, yy, Z_proba, levels=[0.5], colors='black', linewidths=2, linestyles='--')

# Plot training data
scatter = plt.scatter(X_train["feature1"], X_train["feature2"],
                     c=y_train, cmap='RdYlBu', edgecolor="k", s=60, alpha=0.8, label="Train")

# Plot test data
plt.scatter(X_test["feature1"], X_test["feature2"],
           c=y_test, cmap='RdYlBu', marker="^", edgecolor="k", s=100, alpha=0.9, label="Test")

# Custom input
plt.scatter(feature1_input, feature2_input, c="black", marker="X", s=300, label="Your Input")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Naive Bayes: Probability Map and Decision Boundary")
plt.legend()
plt.show()

# 4. Feature distributions by class
plt.figure(figsize=(12, 5))

# Feature 1 distribution
plt.subplot(1, 2, 1)
class0_f1 = df[df["class"] == 0]["feature1"]
class1_f1 = df[df["class"] == 1]["feature1"]

plt.hist(class0_f1, alpha=0.7, label="Class 0", bins=20, color='blue', density=True)
plt.hist(class1_f1, alpha=0.7, label="Class 1", bins=20, color='red', density=True)
plt.axvline(feature1_input, color='black', linestyle='--', linewidth=2, label='Your Input')
plt.xlabel("Feature 1")
plt.ylabel("Density")
plt.title("Feature 1 Distribution by Class")
plt.legend()

# Feature 2 distribution
plt.subplot(1, 2, 2)
class0_f2 = df[df["class"] == 0]["feature2"]
class1_f2 = df[df["class"] == 1]["feature2"]

plt.hist(class0_f2, alpha=0.7, label="Class 0", bins=20, color='blue', density=True)
plt.hist(class1_f2, alpha=0.7, label="Class 1", bins=20, color='red', density=True)
plt.axvline(feature2_input, color='black', linestyle='--', linewidth=2, label='Your Input')
plt.xlabel("Feature 2")
plt.ylabel("Density")
plt.title("Feature 2 Distribution by Class")
plt.legend()

plt.tight_layout()
plt.show()

# 5. Model parameters (means and variances)
print("\n=== Naive Bayes Model Parameters ===")
for class_idx in range(len(model.classes_)):
    print(f"\nClass {model.classes_[class_idx]}:")
    print(f"  Prior probability: {model.class_prior_[class_idx]:.4f}")
    print(f"  Feature means: {model.theta_[class_idx]}")
    print(f"  Feature variances: {model.var_[class_idx]}")