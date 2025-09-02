import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

#knn automatically detects that he has to do OvR is their is a case like multi class classification like in this project as banana apple and mango 1 2 3 so have automatically decided to use OvR rather than explicitly mentioning it to do soo if its logistic classification then we have to do this applying of OvR ecplicitly.

np.random.seed(42)


apple_size = np.random.randint(6, 9, 30)
apple_weight = np.random.randint(40, 70, 30)
apple_label = [0]*30


mango_size = np.random.randint(8, 12, 30)
mango_weight = np.random.randint(60, 90, 30)
mango_label = [1]*30


banana_size = np.random.randint(12, 16, 30)
banana_weight = np.random.randint(85, 120, 30)
banana_label = [2]*30


size = np.concatenate([apple_size, mango_size, banana_size])
weight = np.concatenate([apple_weight, mango_weight, banana_weight])
fruit = np.concatenate([apple_label, mango_label, banana_label])

df = pd.DataFrame({
    "size": size,
    "weight": weight,
    "fruit": fruit
})



X = df[["size", "weight"]]
y = df["fruit"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)


print("Enter fruit size:")
size = float(input())
print("Enter fruit weight:")
weight = float(input())

custom_prediction = model.predict([[size, weight]])[0]

fruit_names = {0: "Apple 🍎", 1: "Mango 🥭", 2: "Banana 🍌"}
print(f"\nThis is likely a {fruit_names[custom_prediction]}")


print("\nAccuracy:", accuracy_score(y_test, predictions))
print("Precision (macro):", precision_score(y_test, predictions, average="macro"))
print("Recall (macro):", recall_score(y_test, predictions, average="macro"))
print("F1 Score (macro):", f1_score(y_test, predictions, average="macro"))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions, target_names=fruit_names.values()))


colors = {0: "red", 1: "green", 2: "yellow"}
labels = {0: "Apple 🍎", 1: "Mango 🥭", 2: "Banana 🍌"}

plt.figure(figsize=(8,6))


for fruit in np.unique(y):
    plt.scatter(
        X_train[y_train==fruit]["size"],
        X_train[y_train==fruit]["weight"],
        c=colors[fruit],
        label=f"{labels[fruit]} (Train)",
        marker="o",
        edgecolor="k"
    )


plt.scatter(X_test["size"], X_test["weight"], c=[colors[i] for i in y_test], 
            marker="*", s=150, label="Test Data")


plt.scatter(size, weight, c=colors[custom_prediction], marker="X", s=200, label="Your Input")

plt.xlabel("Size")
plt.ylabel("Weight")
plt.title("KNN Classification: Fruit Classifier")
plt.legend()
plt.show()
