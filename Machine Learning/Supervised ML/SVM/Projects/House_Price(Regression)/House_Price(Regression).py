import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load data
data = pd.read_csv('house_price_dataset.csv')
print(data.head())

copyData = data.copy()

# Features and target
X = copyData[['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'location_score']]
y = copyData['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVR model
model = SVR(kernel='rbf', C=1000, gamma='scale')
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print metrics
print("=== House Price Prediction Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter house details for price prediction:")
size_sqft = float(input("Enter house size in sqft (800-4000): "))
bedrooms = int(input("Enter number of bedrooms (1-5): "))
bathrooms = float(input("Enter number of bathrooms (1-3.5): "))
age_years = int(input("Enter house age in years (0-50): "))
location_score = float(input("Enter location score (1-10): "))

custom_data = pd.DataFrame({
    'size_sqft': [size_sqft],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'age_years': [age_years],
    'location_score': [location_score]
})

custom_prediction = model.predict(custom_data)
print(f"Predicted House Price: ${custom_prediction[0]:,.2f}")

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

plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("House Price Prediction: Actual vs Predicted")
plt.legend()
plt.savefig("actual_vs_pred.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Feature Importance Plot (using coefficients magnitude - approximation for SVR)
# Since SVR doesn't have direct feature importance, we'll show correlations
correlations = np.abs(copyData[['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'location_score']].corrwith(copyData['price']))

plt.figure(figsize=(8,6))
correlations.plot(kind='bar', color='skyblue')
plt.title("Feature Correlation with House Price")
plt.ylabel("Absolute Correlation")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_correlations.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Residuals Plot
residuals = y_test - predictions
plt.figure(figsize=(8,6))
plt.scatter(predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.title("House Price Prediction: Residuals Plot")
plt.savefig("residuals.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Price Distribution by Key Features
plt.figure(figsize=(12, 8))

# Price vs Size
plt.subplot(2, 2, 1)
plt.scatter(copyData['size_sqft'], copyData['price'], alpha=0.6, color='blue')
plt.scatter(size_sqft, custom_prediction, color='red', s=100, marker='X', label='Your Input')
plt.xlabel("Size (sqft)")
plt.ylabel("Price ($)")
plt.title("Price vs Size")
plt.legend()

# Price vs Bedrooms
plt.subplot(2, 2, 2)
bedroom_groups = copyData.groupby('bedrooms')['price'].mean()
plt.bar(bedroom_groups.index, bedroom_groups.values, alpha=0.7, color='green')
plt.xlabel("Number of Bedrooms")
plt.ylabel("Average Price ($)")
plt.title("Average Price by Bedrooms")

# Price vs Age
plt.subplot(2, 2, 3)
plt.scatter(copyData['age_years'], copyData['price'], alpha=0.6, color='orange')
plt.scatter(age_years, custom_prediction, color='red', s=100, marker='X', label='Your Input')
plt.xlabel("Age (years)")
plt.ylabel("Price ($)")
plt.title("Price vs Age")
plt.legend()

# Price vs Location Score
plt.subplot(2, 2, 4)
plt.scatter(copyData['location_score'], copyData['price'], alpha=0.6, color='purple')
plt.scatter(location_score, custom_prediction, color='red', s=100, marker='X', label='Your Input')
plt.xlabel("Location Score")
plt.ylabel("Price ($)")
plt.title("Price vs Location Score")
plt.legend()

plt.tight_layout()
plt.savefig("price_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nPlots saved as PNG files in the current directory!")