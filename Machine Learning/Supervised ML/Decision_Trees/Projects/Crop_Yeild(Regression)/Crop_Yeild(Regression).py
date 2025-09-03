import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error 
from sklearn.tree import plot_tree

data=pd.read_csv('crop_yield_dataset.csv')
print(data)

copyData=data.copy()


X=copyData[['rainfall_mm','fertilizer_kg','avg_temperature','soil_quality','pesticide_use']]
y=copyData[['yield_tons_per_hectare']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(
    max_depth=5,      
    min_samples_split=10, 
    min_samples_leaf=5, 
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)


print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

print("enter the rainfall in mm:")
rainfall = float(input())
print("enter the fertilizer in kg:")
fertilizer = float(input())
print("enter the average temperature in °C:")
avg_temperature = float(input())
print("enter the soil quality (1-10):")
soil_quality = float(input())
print("enter the pesticide use (1-5):")
pesticide_use = float(input())

custom_data = pd.DataFrame({
    'rainfall_mm': [rainfall],
    'fertilizer_kg': [fertilizer],
    'avg_temperature': [avg_temperature],
    'soil_quality': [soil_quality],
    'pesticide_use': [pesticide_use]
})

custom_prediction = model.predict(custom_data)
print("Predicted Crop Yield (tons per hectare):", custom_prediction[0])


tolerance = 0.1 * y_test.values.flatten()  

correct = np.abs(predictions - y_test.values.flatten()) <= tolerance
wrong = ~correct  

plt.figure(figsize=(7,7))
plt.scatter(y_test[correct], predictions[correct], color='green', label="Correct ✅", alpha=0.7)
plt.scatter(y_test[wrong], predictions[wrong], color='red', label="Wrong ❌", alpha=0.7)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Decision Tree: Correct vs Wrong Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--')  # Perfect prediction line
plt.legend()
plt.show()
# Custom prediction point
plt.scatter(y_test, predictions, alpha=0.6, label="Test Data")
plt.scatter(custom_prediction, custom_prediction, color='purple', marker='X', s=200, label="Your Input Prediction")

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Crop Yield Prediction with Custom Input")
plt.legend()
plt.show()



plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, filled=True)
plt.title("Decision Tree Visualization")
plt.show()
