import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import random

# Generate synthetic dataset
random.seed(42)
np.random.seed(42)

data = {"feature1": [], "feature2": [], "target": []}

for i in range(600):
    feature1 = np.random.uniform(0, 10)
    feature2 = np.random.uniform(0, 10)
    target = np.sin(feature1) * 3 + np.cos(feature2) * 2 + feature1 * 0.5 + np.random.normal(0, 0.5)
    
    data["feature1"].append(feature1)
    data["feature2"].append(feature2)
    data["target"].append(target)

df = pd.DataFrame(data)
X = df[["feature1", "feature2"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVR model
model = SVR(kernel='rbf', C=100, gamma='scale')
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

print("=== SVR Regression Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

# Custom prediction
print("\n=== Custom Prediction ===")
f1 = float(input("Enter feature1 (0-10): "))
f2 = float(input("Enter feature2 (0-10): "))

custom_data = np.array([[f1, f2]])
custom_data_scaled = scaler.transform(custom_data)
custom_prediction = model.predict(custom_data_scaled)
print(f"Predicted value: {custom_prediction[0]:.2f}")

# Plot
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
plt.title("SVR: Actual vs Predicted")
plt.legend()
plt.show()