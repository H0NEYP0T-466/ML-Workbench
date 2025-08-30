import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

data = {
    "studyHours": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30
    ],
    "passExam": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}



df = pd.DataFrame(data)
X = df[["studyHours"]]
y = df["passExam"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

hours = float(input("Enter the number of study hours: "))


model = LogisticRegression()
model.fit(X_train, y_train)


predicted_pass = model.predict(pd.DataFrame([[hours]], columns=["studyHours"]))
print(f"Predicted pass status for {hours} study hours: {predicted_pass[0]}")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.scatter(X, y, color="blue", label="Actual Data")         
plt.plot(X,model.predict(X), color="red", label="Regression Line") 


plt.scatter(hours, predicted_pass, color="green", s=100, label="Predicted Point")

plt.xlabel("Study Hours")
plt.ylabel("Pass Exam")
plt.title("Logistic Regression - Study Hours vs Pass Exam")
plt.legend()
plt.show()
