import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('car_price_dataset.csv')
copyData = data.copy()

# Features and target
X = copyData[['mileage', 'age', 'engine_size', 'horsepower', 'fuel_efficiency']]
y = copyData['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge model
model = Ridge(alpha=100.0, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Print metrics
print("=== Car Price Prediction Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

# Get user input
print("\n=== Custom Prediction ===")
mileage = float(input("Enter mileage (10000-200000): "))
age = int(input("Enter age in years (0-15): "))
engine_size = float(input("Enter engine size in L (1.0-6.0): "))
horsepower = int(input("Enter horsepower (100-500): "))
fuel_efficiency = float(input("Enter fuel efficiency MPG (15-45): "))

custom_data = np.array([[mileage, age, engine_size, horsepower, fuel_efficiency]])
custom_data_scaled = scaler.transform(custom_data)
custom_prediction = model.predict(custom_data_scaled)

print(f"Predicted Car Price: ${custom_prediction[0]:,.2f}")

# PLOTS
tolerance = 0.1 * np.abs(y_test.values)
correct = np.abs(predictions - y_test.values) <= tolerance
wrong = ~correct

plt.figure(figsize=(8,8))
plt.scatter(y_test[correct], predictions[correct], color='green', 
           label=f"Correct ✅ ({correct.sum()})", alpha=0.7)
plt.scatter(y_test[wrong], predictions[wrong], color='red', 
           label=f"Wrong ❌ ({wrong.sum()})", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', label='Perfect Prediction')
plt.scatter(custom_prediction, custom_prediction, color='black', marker='X', s=200, label="Your Input ⭐")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Ridge Regression: Car Price Prediction")
plt.legend()
plt.savefig("actual_vs_pred.png", dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as actual_vs_pred.png")