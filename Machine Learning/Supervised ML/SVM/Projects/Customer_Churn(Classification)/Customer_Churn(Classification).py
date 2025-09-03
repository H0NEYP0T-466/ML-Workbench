import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# Load data
data = pd.read_csv("customer_churn_dataset.csv")

copyData = data.copy()

# Features and target
X = copyData[["monthly_charges", "total_charges", "tenure_months", "customer_service_calls", "data_usage_gb"]]
copyData["churn"] = copyData["churn"].map({"No": 0, "Yes": 1})
y = copyData["churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print metrics
print("=== Customer Churn Prediction Results ===")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter customer details for churn prediction:")
monthly_charges = float(input("Enter monthly charges ($30-120): "))
total_charges = float(input("Enter total charges: "))
tenure_months = int(input("Enter tenure in months (1-72): "))
service_calls = int(input("Enter customer service calls (0-8): "))
data_usage = float(input("Enter data usage in GB (0.5-50): "))

custom_data = pd.DataFrame({
    "monthly_charges": [monthly_charges],
    "total_charges": [total_charges],
    "tenure_months": [tenure_months],
    "customer_service_calls": [service_calls],
    "data_usage_gb": [data_usage]
})

custom_prediction = model.predict(custom_data)
if custom_prediction == 1:
    print("Customer is likely to CHURN ⚠️")
else:
    print("Customer is likely to STAY 🎉")

# PLOTS

# 1. Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Customer Churn")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Correct vs Wrong Predictions Scatter Plot (using first two features)
correct_idx = predictions == y_test
wrong_idx = predictions != y_test

plt.figure(figsize=(8,6))

# Correct predictions
plt.scatter(X_test.loc[correct_idx, "monthly_charges"], 
            X_test.loc[correct_idx, "tenure_months"], 
            color="green", alpha=0.6, label=f"Correct ✅ ({correct_idx.sum()})")

# Wrong predictions
plt.scatter(X_test.loc[wrong_idx, "monthly_charges"], 
            X_test.loc[wrong_idx, "tenure_months"], 
            color="red", alpha=0.6, label=f"Wrong ❌ ({wrong_idx.sum()})")

# Custom input
plt.scatter(monthly_charges, tenure_months, color="black", marker="X", s=200, label="Your Input ⭐")

plt.xlabel("Monthly Charges ($)")
plt.ylabel("Tenure (Months)")
plt.title("Customer Churn: Correct vs Wrong Predictions")
plt.legend()
plt.savefig("correct_vs_wrong.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Feature Distribution Plot
plt.figure(figsize=(12, 8))

features = ["monthly_charges", "tenure_months", "customer_service_calls", "data_usage_gb"]
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    
    # Split by churn status
    stay_data = copyData[copyData["churn"] == 0][feature]
    churn_data = copyData[copyData["churn"] == 1][feature]
    
    plt.hist(stay_data, alpha=0.7, label="Stay", bins=20, color='green')
    plt.hist(churn_data, alpha=0.7, label="Churn", bins=20, color='red')
    plt.xlabel(feature.replace('_', ' ').title())
    plt.ylabel("Frequency")
    plt.legend()

plt.suptitle("Feature Distributions by Churn Status")
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nPlots saved as PNG files in the current directory!")