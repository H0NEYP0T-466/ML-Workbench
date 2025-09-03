import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load data
data = pd.read_csv('energy_consumption_dataset.csv')
print(data.head())

copyData = data.copy()

# Features and target
X = copyData[['temperature', 'humidity', 'wind_speed', 'solar_radiation', 'hour_of_day']]
y = copyData['energy_consumption']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print metrics
print("=== Energy Consumption Prediction Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

# Feature importance
feature_importance = model.feature_importances_
feature_names = ['temperature', 'humidity', 'wind_speed', 'solar_radiation', 'hour_of_day']

print("\nFeature Importances:")
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter environmental conditions for energy consumption prediction:")
temperature = float(input("Enter temperature in °C (-10 to 40): "))
humidity = float(input("Enter humidity percentage (20-90): "))
wind_speed = float(input("Enter wind speed in m/s (0-20): "))
solar_radiation = float(input("Enter solar radiation in W/m² (0-1000): "))
hour_of_day = int(input("Enter hour of day (0-23): "))

custom_data = pd.DataFrame({
    'temperature': [temperature],
    'humidity': [humidity],
    'wind_speed': [wind_speed],
    'solar_radiation': [solar_radiation],
    'hour_of_day': [hour_of_day]
})

custom_prediction = model.predict(custom_data)
print(f"Predicted Energy Consumption: {custom_prediction[0]:.2f} kWh")

# PLOTS

# 1. Feature Importance
plt.figure(figsize=(10,6))
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=True)

plt.barh(importance_df['feature'], importance_df['importance'], color='lightgreen', edgecolor='black')
plt.xlabel("Importance")
plt.title("Random Forest: Feature Importance for Energy Consumption")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
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

plt.xlabel("Actual Energy Consumption (kWh)")
plt.ylabel("Predicted Energy Consumption (kWh)")
plt.title("Energy Consumption: Actual vs Predicted")
plt.legend()
plt.savefig("actual_vs_pred.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Feature Impact Analysis
plt.figure(figsize=(15, 10))

# Temperature vs Energy
plt.subplot(2, 3, 1)
plt.scatter(copyData['temperature'], copyData['energy_consumption'], alpha=0.6, color='red')
plt.scatter(temperature, custom_prediction, color='black', s=100, marker='X', label='Your Input')
plt.xlabel("Temperature (°C)")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Temperature vs Energy Consumption")
plt.legend()

# Humidity vs Energy
plt.subplot(2, 3, 2)
plt.scatter(copyData['humidity'], copyData['energy_consumption'], alpha=0.6, color='blue')
plt.scatter(humidity, custom_prediction, color='black', s=100, marker='X', label='Your Input')
plt.xlabel("Humidity (%)")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Humidity vs Energy Consumption")
plt.legend()

# Wind Speed vs Energy
plt.subplot(2, 3, 3)
plt.scatter(copyData['wind_speed'], copyData['energy_consumption'], alpha=0.6, color='green')
plt.scatter(wind_speed, custom_prediction, color='black', s=100, marker='X', label='Your Input')
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Wind Speed vs Energy Consumption")
plt.legend()

# Solar Radiation vs Energy
plt.subplot(2, 3, 4)
plt.scatter(copyData['solar_radiation'], copyData['energy_consumption'], alpha=0.6, color='orange')
plt.scatter(solar_radiation, custom_prediction, color='black', s=100, marker='X', label='Your Input')
plt.xlabel("Solar Radiation (W/m²)")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Solar Radiation vs Energy Consumption")
plt.legend()

# Hour of Day vs Energy (box plot)
plt.subplot(2, 3, 5)
hourly_consumption = [copyData[copyData['hour_of_day'] == h]['energy_consumption'].values for h in range(24)]
plt.boxplot(hourly_consumption, positions=range(24))
plt.scatter(hour_of_day, custom_prediction, color='red', s=100, marker='X', label='Your Input')
plt.xlabel("Hour of Day")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Hourly Energy Consumption Pattern")
plt.xticks(range(0, 24, 4))
plt.legend()

# Residuals
plt.subplot(2, 3, 6)
residuals = y_test - predictions
plt.scatter(predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Energy (kWh)")
plt.ylabel("Residuals (kWh)")
plt.title("Residuals Plot")

plt.tight_layout()
plt.savefig("energy_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Seasonal/Time Pattern Analysis
plt.figure(figsize=(12, 8))

# Daily pattern
plt.subplot(2, 2, 1)
hourly_avg = copyData.groupby('hour_of_day')['energy_consumption'].mean()
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, color='blue')
plt.scatter(hour_of_day, custom_prediction, color='red', s=100, marker='X', label='Your Input')
plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Daily Energy Consumption Pattern")
plt.grid(True, alpha=0.3)
plt.legend()

# Temperature bins
plt.subplot(2, 2, 2)
temp_bins = pd.cut(copyData['temperature'], bins=8)
temp_avg = copyData.groupby(temp_bins)['energy_consumption'].mean()
temp_centers = [interval.mid for interval in temp_avg.index]
plt.bar(range(len(temp_centers)), temp_avg.values, alpha=0.7, color='red')
plt.xlabel("Temperature Range")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Energy Consumption by Temperature Range")
plt.xticks(range(len(temp_centers)), [f"{center:.1f}" for center in temp_centers], rotation=45)

# Feature correlations heatmap
plt.subplot(2, 2, 3)
correlation_matrix = copyData[feature_names + ['energy_consumption']].corr()
import seaborn as sns
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title("Feature Correlation Matrix")

# Your input summary
plt.subplot(2, 2, 4)
input_values = [temperature, humidity, wind_speed, solar_radiation, hour_of_day]
plt.bar(range(len(feature_names)), input_values, color=['red', 'blue', 'green', 'orange', 'purple'], alpha=0.7)
plt.xlabel("Features")
plt.ylabel("Values")
plt.title(f"Your Input Profile\nPredicted: {custom_prediction[0]:.1f} kWh")
plt.xticks(range(len(feature_names)), feature_names, rotation=45)

plt.tight_layout()
plt.savefig("pattern_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nPlots saved as PNG files in the current directory!")