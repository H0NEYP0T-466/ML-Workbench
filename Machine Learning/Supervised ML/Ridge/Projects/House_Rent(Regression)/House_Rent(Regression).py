import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('house_rent_dataset.csv')
X = data[['area', 'bedrooms', 'bathrooms', 'location_score']]
y = data['rent']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = Ridge(alpha=50.0, random_state=42)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

print("=== House Rent Prediction Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

# Custom prediction
print("\n=== Custom Prediction ===")
area = float(input("Enter house area (400-2000 sqft): "))
bedrooms = int(input("Enter bedrooms (1-4): "))
bathrooms = int(input("Enter bathrooms (1-3): "))
location_score = float(input("Enter location score (1-10): "))

custom_data = np.array([[area, bedrooms, bathrooms, location_score]])
custom_data_scaled = scaler.transform(custom_data)
custom_prediction = model.predict(custom_data_scaled)
print(f"Predicted Monthly Rent: ${custom_prediction[0]:,.2f}")

# Plot
tolerance = 0.1 * np.abs(y_test.values)
correct = np.abs(predictions - y_test.values) <= tolerance
wrong = ~correct

plt.figure(figsize=(8,8))
plt.scatter(y_test[correct], predictions[correct], color='green', label=f"Correct ✅ ({correct.sum()})", alpha=0.7)
plt.scatter(y_test[wrong], predictions[wrong], color='red', label=f"Wrong ❌ ({wrong.sum()})", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', label='Perfect Prediction')
plt.scatter(custom_prediction, custom_prediction, color='black', marker='X', s=200, label="Your Input ⭐")
plt.xlabel("Actual Rent ($)")
plt.ylabel("Predicted Rent ($)")
plt.title("Ridge Regression: House Rent Prediction")
plt.legend()
plt.savefig("actual_vs_pred.png", dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved!")