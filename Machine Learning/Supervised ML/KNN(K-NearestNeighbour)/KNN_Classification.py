import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

data = {
    "size": [8, 6, 8, 8, 6, 7, 7, 8, 6, 7, 7, 8, 6, 6, 8, 7, 7, 6, 8, 7, 7, 7, 7, 6, 8, 
             11, 12, 12, 12, 12, 9, 12, 9, 9, 11, 11, 11, 11, 11, 12, 9, 12, 9, 9, 12, 12, 12, 12, 10, 12],
    "weight": [43, 36, 54, 49, 41, 53, 40, 42, 46, 52, 37, 50, 44, 41, 46, 40, 41, 52, 44, 46, 49, 53, 42, 54, 51,
               78, 72, 73, 88, 72, 85, 62, 86, 83, 78, 74, 88, 67, 77, 70, 75, 83, 79, 69, 85, 63, 65, 87, 75, 84],
    "fruit": [0]*25 + [1]*25  
}

# 0 is apple
# 1 is mango
df = pd.DataFrame(data)

X = df[['size', 'weight']]
y = df['fruit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)


print("Enter size:")
size = float(input())   
print("Enter weight:")
weight = float(input())

custom_prediction = model.predict([[size, weight]])
if custom_prediction[0] == 0:
    print("This is likely an Apple 🍎")
else:
    print("This is likely a Mango 🥭")

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))


h = 0.1  


x_min, x_max = X['size'].min() - 1, X['size'].max() + 1
y_min, y_max = X['weight'].min() - 1, X['weight'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)


plt.scatter(X_train['size'], X_train['weight'], c=y_train, 
            cmap=plt.cm.RdYlBu, edgecolor='k', s=100, label="Train")
plt.scatter(X_test['size'], X_test['weight'], c=y_test, 
            cmap=plt.cm.RdYlBu, edgecolor='k', marker='*', s=200, label="Test")


plt.scatter(size, weight, c='black', marker='X', s=300, label="Your Input")

plt.xlabel("Size")
plt.ylabel("Weight")
plt.title("KNN Decision Boundary (Apples vs Mangoes)")
plt.legend()
plt.show()

