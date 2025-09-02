import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Assignment": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    "marks": [20, 25, 35, 40, 50, 60, 65, 70, 85, 90]
}

df = pd.DataFrame(data)

X = df[["hours", "Assignment"]]
y = df["marks"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)


print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))

print("Enter study hours:")
hours = float(input())
print("Enter assignment number:")
assignment = float(input())
predicted_marks = model.predict([[hours, assignment]])
print(f"Predicted Marks: {predicted_marks[0]:.2f}")




fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X["hours"], X["Assignment"], y, color="blue", label="Actual Data")


ax.scatter(X_test["hours"], X_test["Assignment"], y_test, 
           color="green", s=100, marker="*", label="Test Data")


ax.scatter(hours, assignment, predicted_marks, 
           color="black", marker="X", s=200, label="Your Input")


ax.set_xlabel("Study Hours")
ax.set_ylabel("Assignments")
ax.set_zlabel("Marks")
ax.set_title("KNN Regression: Study Hours + Assignments vs Marks")

plt.legend()
plt.show()

