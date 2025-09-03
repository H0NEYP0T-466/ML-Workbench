import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random

# Generate synthetic dataset
random.seed(42)
np.random.seed(42)

data = {
    "feature1": [],
    "feature2": [],
    "target": []
}

# Generate 500+ data points with realistic relationships
for i in range(600):
    feature1 = np.random.uniform(0, 10)
    feature2 = np.random.uniform(0, 10)
    
    # Create a non-linear relationship with noise
    target = 2 * feature1 + 1.5 * feature2 + 0.5 * feature1 * feature2 + np.random.normal(0, 2)
    
    data["feature1"].append(feature1)
    data["feature2"].append(feature2)
    data["target"].append(target)

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["feature1", "feature2"]]
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVR model
model = SVR(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print metrics
print("=== SVR Regression Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter values for prediction:")
feature1_input = float(input("Enter feature1 value (0-10): "))
feature2_input = float(input("Enter feature2 value (0-10): "))

custom_data = pd.DataFrame({
    "feature1": [feature1_input],
    "feature2": [feature2_input]
})

custom_prediction = model.predict(custom_data)
print(f"Predicted target value: {custom_prediction[0]:.2f}")

# PLOTS

# 1. Actual vs Predicted with tolerance
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
plt.title("SVR: Actual vs Predicted Values")
plt.legend()
plt.show()

# 2. Residuals plot
residuals = y_test - predictions
plt.figure(figsize=(8,6))
plt.scatter(predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("SVR: Residuals Plot")
plt.show()

# 3. Feature importance visualization (showing prediction surface for 2D)
# Create a mesh grid for visualization
x1_range = np.linspace(X["feature1"].min(), X["feature1"].max(), 50)
x2_range = np.linspace(X["feature2"].min(), X["feature2"].max(), 50)
xx1, xx2 = np.meshgrid(x1_range, x2_range)

# Predict on the mesh grid
grid_points = np.c_[xx1.ravel(), xx2.ravel()]
zz = model.predict(grid_points).reshape(xx1.shape)

# 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(xx1, xx2, zz, alpha=0.6, cmap='viridis')

# Plot the training points
ax.scatter(X_train["feature1"], X_train["feature2"], y_train, 
          color='red', s=20, alpha=0.6, label='Training Data')

# Plot custom input
ax.scatter(feature1_input, feature2_input, custom_prediction, 
          color='black', s=100, marker='X', label='Your Input')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('SVR Prediction Surface')
plt.colorbar(surf)
plt.legend()
plt.show()