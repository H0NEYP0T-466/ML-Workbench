import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "assignments_done": [1, 1, 2, 3, 3, 4, 5, 6, 7, 8],
    "result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass"]
}

df = pd.DataFrame(data)

X = df[["study_hours", "assignments_done"]]
df["result"] = df["result"].map({"Fail": 0, "Pass": 1})
y = df["result"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))



hours = float(input("Enter study hours: "))
assignments = int(input("Enter assignments completed: "))
prediction = model.predict([[hours, assignments]])
if prediction[0] == 1:
    print("The student is likely to Pass 🎉")
else:
    print("The student is likely to Fail 😞")




h = 0.1  

x_min, x_max = X["study_hours"].min() - 1, X["study_hours"].max() + 1
y_min, y_max = X["assignments_done"].min() - 1, X["assignments_done"].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.4)


plt.scatter(X_train["study_hours"], X_train["assignments_done"],
            c=y_train, cmap=plt.cm.RdYlBu, edgecolor="k", s=100, label="Train")


plt.scatter(X_test["study_hours"], X_test["assignments_done"],
            c=y_test, cmap=plt.cm.RdYlBu, marker="*", edgecolor="k", s=200, label="Test")

plt.scatter(hours, assignments, c="black", marker="X", s=300, label="Your Input")

plt.xlabel("Study Hours")
plt.ylabel("Assignments Done")
plt.title("Decision Tree Classification: Pass vs Fail")
plt.legend()
plt.show()
