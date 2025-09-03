import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
    # Create multiple clusters with some noise
    if i < 200:
        # Class 0 - cluster 1
        feature1 = np.random.normal(2, 1.2)
        feature2 = np.random.normal(2, 1.2)
        class_label = 0
    elif i < 400:
        # Class 0 - cluster 2
        feature1 = np.random.normal(7, 1.0)
        feature2 = np.random.normal(3, 1.0)
        class_label = 0
    else:
        # Class 1 - main cluster
        feature1 = np.random.normal(4, 1.5)
        feature2 = np.random.normal(6, 1.5)
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

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print metrics
print("=== Random Forest Classification Results ===")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Feature importance
feature_importance = model.feature_importances_
feature_names = ["feature1", "feature2"]

print("\nFeature Importances:")
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter values for prediction:")
feature1_input = float(input("Enter feature1 value (typically 0-10): "))
feature2_input = float(input("Enter feature2 value (typically 0-10): "))

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
plt.title("Confusion Matrix - Random Forest Classification")
plt.show()

# 2. Feature Importance Bar Plot
plt.figure(figsize=(8,5))
plt.bar(feature_names, feature_importance, color='lightgreen', edgecolor='black')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest: Feature Importance")
plt.show()

# 3. Correct vs Wrong Predictions Scatter Plot
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
plt.title("Random Forest Classification: Correct vs Wrong Predictions")
plt.legend()
plt.show()

# 4. Decision Boundary (2D)
h = 0.1
x_min, x_max = X["feature1"].min() - 1, X["feature1"].max() + 1
y_min, y_max = X["feature2"].min() - 1, X["feature2"].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.4)

# Plot training data
plt.scatter(X_train["feature1"], X_train["feature2"],
            c=y_train, cmap=plt.cm.RdYlBu, edgecolor="k", s=60, label="Train", alpha=0.7)

# Plot test data
plt.scatter(X_test["feature1"], X_test["feature2"],
            c=y_test, cmap=plt.cm.RdYlBu, marker="^", edgecolor="k", s=100, label="Test")

# Custom input
plt.scatter(feature1_input, feature2_input, c="black", marker="X", s=300, label="Your Input")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Random Forest Classification Decision Boundary")
plt.legend()
plt.show()