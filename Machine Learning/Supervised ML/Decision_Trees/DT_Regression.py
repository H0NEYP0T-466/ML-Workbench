import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import random

data = {
    "car_age": [],
    "mileage": [],
    "price": []
}

random.seed(42)  

for age in range(1, 51):  
 
    mileage = age * random.randint(8, 12) + random.randint(0, 10)
    
    price = max(2000, 52000 - (age * random.randint(1800, 2200)) + random.randint(-1000, 1000))
    
    data["car_age"].append(age)
    data["mileage"].append(mileage)
    data["price"].append(price)



df = pd.DataFrame(data)

X = df[["car_age", "mileage"]]
y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


reg = DecisionTreeRegressor(max_depth=6, random_state=42)
reg.fit(X_train, y_train)


y_pred = reg.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))


age = float(input("Enter car age (years): "))
miles = float(input("Enter mileage (in thousands): "))
prediction = reg.predict([[age, miles]])
print(f"Predicted Price: ${prediction[0]:,.2f}")


plt.figure(figsize=(10, 6))
plot_tree(reg, feature_names=["car_age", "mileage"], filled=True)
plt.show()
