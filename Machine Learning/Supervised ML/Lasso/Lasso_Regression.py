import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import random

# Generate synthetic dataset
random.seed(42)
np.random.seed(42)

data = {"feature1": [], "feature2": [], "feature3": [], "feature4": [], "feature5": [], "target": []}

# Generate 600 data points with some irrelevant features for Lasso to select
for i in range(600):
    feature1 = np.random.uniform(0, 10)
    feature2 = np.random.uniform(0, 10)
    feature3 = np.random.uniform(0, 10)  # Irrelevant feature
    feature4 = np.random.uniform(0, 10)
    feature5 = np.random.uniform(0, 10)  # Irrelevant feature
    
    # Only feature1, feature2, and feature4 are truly predictive
    target = 2 * feature1 + 3 * feature2 + 1.5 * feature4 + np.random.normal(0, 1)
    
    data["feature1"].append(feature1)
    data["feature2"].append(feature2)
    data["feature3"].append(feature3)
    data["feature4"].append(feature4)
    data["feature5"].append(feature5)
    data["target"].append(target)

df = pd.DataFrame(data)
X = df[["feature1", "feature2", "feature3", "feature4", "feature5"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Lasso model
model = Lasso(alpha=0.1, random_state=42)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

print("=== Lasso Regression Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]
print("\nLasso Coefficients (Feature Selection):")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")

# Custom prediction
print("\n=== Custom Prediction ===")
f1 = float(input("Enter feature1 (0-10): "))
f2 = float(input("Enter feature2 (0-10): "))
f3 = float(input("Enter feature3 (0-10): "))
f4 = float(input("Enter feature4 (0-10): "))
f5 = float(input("Enter feature5 (0-10): "))

custom_data = np.array([[f1, f2, f3, f4, f5]])
custom_data_scaled = scaler.transform(custom_data)
custom_prediction = model.predict(custom_data_scaled)
print(f"Predicted value: {custom_prediction[0]:.2f}")

# Plots
plt.figure(figsize=(8,5))
plt.bar(feature_names, model.coef_, color=['red' if abs(c) < 0.1 else 'blue' for c in model.coef_])
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.title("Lasso Regression: Feature Selection (Red = Nearly Zero)")
plt.show()

tolerance = 0.1 * np.abs(y_test.values)
correct = np.abs(predictions - y_test.values) <= tolerance
wrong = ~correct

plt.figure(figsize=(8,8))
plt.scatter(y_test[correct], predictions[correct], color='green', label=f"Correct ✅ ({correct.sum()})", alpha=0.7)
plt.scatter(y_test[wrong], predictions[wrong], color='red', label=f"Wrong ❌ ({wrong.sum()})", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', label='Perfect Prediction')
plt.scatter(custom_prediction, custom_prediction, color='black', marker='X', s=200, label="Your Input ⭐")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Lasso Regression: Actual vs Predicted")
plt.legend()
plt.show()