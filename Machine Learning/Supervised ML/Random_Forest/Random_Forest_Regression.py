import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random

# Generate synthetic dataset
random.seed(42)
np.random.seed(42)

data = {
    "feature1": [],
    "feature2": [],
    "feature3": [],
    "target": []
}

# Generate 500+ data points with realistic relationships
for i in range(600):
    feature1 = np.random.uniform(0, 10)
    feature2 = np.random.uniform(0, 10)
    feature3 = np.random.uniform(0, 10)
    
    # Create a complex non-linear relationship with noise
    target = (2 * feature1 + 
              1.5 * feature2 + 
              0.8 * feature3 + 
              0.3 * feature1 * feature2 + 
              0.2 * feature2 * feature3 + 
              np.sin(feature1) * 2 +
              np.random.normal(0, 1))
    
    data["feature1"].append(feature1)
    data["feature2"].append(feature2)
    data["feature3"].append(feature3)
    data["target"].append(target)

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["feature1", "feature2", "feature3"]]
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print metrics
print("=== Random Forest Regression Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

# Feature importance
feature_importance = model.feature_importances_
feature_names = ["feature1", "feature2", "feature3"]

print("\nFeature Importances:")
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter values for prediction:")
feature1_input = float(input("Enter feature1 value (0-10): "))
feature2_input = float(input("Enter feature2 value (0-10): "))
feature3_input = float(input("Enter feature3 value (0-10): "))

custom_data = pd.DataFrame({
    "feature1": [feature1_input],
    "feature2": [feature2_input],
    "feature3": [feature3_input]
})

custom_prediction = model.predict(custom_data)
print(f"Predicted target value: {custom_prediction[0]:.2f}")

# PLOTS

# 1. Feature Importance Bar Plot
plt.figure(figsize=(8,5))
plt.bar(feature_names, feature_importance, color='lightcoral', edgecolor='black')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Regression: Feature Importance")
plt.show()

# 2. Actual vs Predicted with tolerance
tolerance = 0.1 * np.abs(y_test.values)  # 10% tolerance
correct = np.abs(predictions - y_test.values) <= tolerance
wrong = ~correct

plt.figure(figsize=(8,8))
plt.scatter(y_test[correct], predictions[correct], color='green', 
           label=f"Correct ✅ ({correct.sum()})", alpha=0.7)
plt.scatter(y_test[wrong], predictions[wrong], color='red', 
           label=f"Wrong ❌ ({wrong.sum()})", alpha=0.7)

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', label='Perfect Prediction')

# Custom prediction point
plt.scatter(custom_prediction, custom_prediction, color='black', marker='X', s=200, label="Your Input ⭐")

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression: Actual vs Predicted Values")
plt.legend()
plt.show()

# 3. Residuals plot
residuals = y_test - predictions
plt.figure(figsize=(8,6))
plt.scatter(predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Random Forest Regression: Residuals Plot")
plt.show()

# 4. Individual tree predictions (first 5 trees)
plt.figure(figsize=(12, 8))

# Get predictions from individual trees
individual_predictions = []
for i, tree in enumerate(model.estimators_[:5]):  # First 5 trees
    tree_pred = tree.predict(X_test)
    individual_predictions.append(tree_pred)
    
    plt.subplot(2, 3, i+1)
    plt.scatter(y_test, tree_pred, alpha=0.6, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Tree {i+1}")

# Ensemble prediction
plt.subplot(2, 3, 6)
plt.scatter(y_test, predictions, alpha=0.6, s=20, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ensemble (All Trees)")

plt.suptitle("Individual Trees vs Ensemble Predictions")
plt.tight_layout()
plt.show()

# 5. Feature interaction visualization
plt.figure(figsize=(12, 4))

# Feature1 vs Target
plt.subplot(1, 3, 1)
plt.scatter(X_test["feature1"], y_test, alpha=0.6, label="Actual", color='blue')
plt.scatter(X_test["feature1"], predictions, alpha=0.6, label="Predicted", color='red')
plt.scatter(feature1_input, custom_prediction, color='black', s=100, marker='X', label='Your Input')
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.legend()
plt.title("Feature 1 vs Target")

# Feature2 vs Target
plt.subplot(1, 3, 2)
plt.scatter(X_test["feature2"], y_test, alpha=0.6, label="Actual", color='blue')
plt.scatter(X_test["feature2"], predictions, alpha=0.6, label="Predicted", color='red')
plt.scatter(feature2_input, custom_prediction, color='black', s=100, marker='X', label='Your Input')
plt.xlabel("Feature 2")
plt.ylabel("Target")
plt.legend()
plt.title("Feature 2 vs Target")

# Feature3 vs Target
plt.subplot(1, 3, 3)
plt.scatter(X_test["feature3"], y_test, alpha=0.6, label="Actual", color='blue')
plt.scatter(X_test["feature3"], predictions, alpha=0.6, label="Predicted", color='red')
plt.scatter(feature3_input, custom_prediction, color='black', s=100, marker='X', label='Your Input')
plt.xlabel("Feature 3")
plt.ylabel("Target")
plt.legend()
plt.title("Feature 3 vs Target")

plt.tight_layout()
plt.show()