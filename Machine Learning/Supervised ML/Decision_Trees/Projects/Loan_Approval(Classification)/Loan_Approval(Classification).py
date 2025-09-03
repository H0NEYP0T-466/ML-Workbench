import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

data=pd.read_csv("loan_approval_dataset.csv")

copyData=data.copy()

X=copyData[["income","loan_amount","credit_score","employment_years","savings"]]
copyData["loan_status"]=copyData["loan_status"].map({"Approved":1,"Denied":0})
y=copyData["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)
predictions=model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))


print("enter the income:")
income = float(input())
print("enter the loan amount:")
loan_amount = float(input())
print("enter the credit score:")
credit_score = float(input())
print("enter the employment years:")
employment_years = float(input())
print("enter the savings:")
savings = float(input())

custom_data = pd.DataFrame({
    "income": [income],
    "loan_amount": [loan_amount],
    "credit_score": [credit_score],
    "employment_years": [employment_years],
    "savings": [savings]
})

custom_prediction = model.predict(custom_data)
if custom_prediction == 1:
    print("Loan approved")
else:
    print("loan not approved")


    


cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Denied","Approved"], yticklabels=["Denied","Approved"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Loan Approval")
plt.show()

# Identify correct vs wrong predictions
correct_idx = predictions == y_test
wrong_idx = predictions != y_test

plt.figure(figsize=(8,6))

# Correct predictions
plt.scatter(X_test.loc[correct_idx, "income"], 
            X_test.loc[correct_idx, "loan_amount"], 
            color="green", alpha=0.6, label=f"Correct ✅ ({correct_idx.sum()})")

# Wrong predictions
plt.scatter(X_test.loc[wrong_idx, "income"], 
            X_test.loc[wrong_idx, "loan_amount"], 
            color="red", alpha=0.6, label=f"Wrong ❌ ({wrong_idx.sum()})")

# Custom input
plt.scatter(income, loan_amount, color="black", marker="X", s=200, label="Your Input ⭐")

plt.xlabel("Income")
plt.ylabel("Loan Amount")
plt.title("Loan Approval: Correct vs Wrong Predictions")
plt.legend()
plt.show()

