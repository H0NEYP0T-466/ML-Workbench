import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(42)


size = np.random.randint(6, 16, 60)      
weight = np.random.randint(40, 120, 60)  


price = size * 5 + weight * 2 + np.random.randint(-10, 10, 60)

df = pd.DataFrame({
    "size": size,
    "weight": weight,
    "price": price
})

print(df)

X = df[["size", "weight"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Evaluation Metrics:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue", s=80, label="Predicted vs Actual")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", label="Perfect Prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("KNN Regression: Fruit Price Prediction")
plt.legend()
plt.show()

new_size = int(input("Enter fruit size: "))
new_weight = int(input("Enter fruit weight: "))
predicted_price = model.predict([[new_size, new_weight]])
print(f"\n💰 Predicted Price for fruit (Size={new_size}, Weight={new_weight}): {predicted_price[0]:.2f}")
