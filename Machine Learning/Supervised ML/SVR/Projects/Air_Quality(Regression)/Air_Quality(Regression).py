import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('air_quality_dataset.csv')
X = data[['pm25', 'pm10', 'no2', 'traffic']]
y = data['aqi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVR(kernel='rbf', C=100, gamma='scale')
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

print("=== Air Quality Index Prediction Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))

pm25 = float(input("Enter PM2.5 level: "))
pm10 = float(input("Enter PM10 level: "))
no2 = float(input("Enter NO2 level: "))
traffic = float(input("Enter traffic density: "))

custom_data = scaler.transform([[pm25, pm10, no2, traffic]])
prediction = model.predict(custom_data)
print(f"Predicted Air Quality Index: {prediction[0]:.1f}")

tolerance = 0.1 * np.abs(y_test.values)
correct = np.abs(predictions - y_test.values) <= tolerance
wrong = ~correct

plt.figure(figsize=(8,8))
plt.scatter(y_test[correct], predictions[correct], color='green', label=f"Correct ✅ ({correct.sum()})", alpha=0.7)
plt.scatter(y_test[wrong], predictions[wrong], color='red', label=f"Wrong ❌ ({wrong.sum()})", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', label='Perfect Prediction')
plt.scatter(prediction, prediction, color='black', marker='X', s=200, label="Your Input ⭐")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("SVR: Air Quality Index Prediction")
plt.legend()
plt.savefig("actual_vs_pred.png", dpi=300, bbox_inches='tight')
plt.show()
print("Plot saved!")