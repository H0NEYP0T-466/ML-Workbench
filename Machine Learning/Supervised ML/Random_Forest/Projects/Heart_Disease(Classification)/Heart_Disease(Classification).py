import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# Load data
data = pd.read_csv("heart_disease_dataset.csv")

copyData = data.copy()

# Features and target
X = copyData[["age", "cholesterol", "blood_pressure", "heart_rate", "exercise_hours"]]
copyData["heart_disease"] = copyData["heart_disease"].map({"No": 0, "Yes": 1})
y = copyData["heart_disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print metrics
print("=== Heart Disease Prediction Results ===")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Feature importance
feature_importance = model.feature_importances_
feature_names = ["age", "cholesterol", "blood_pressure", "heart_rate", "exercise_hours"]

print("\nFeature Importances:")
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter patient details for heart disease prediction:")
age = int(input("Enter age (25-80): "))
cholesterol = int(input("Enter cholesterol level (150-350 mg/dL): "))
blood_pressure = int(input("Enter systolic blood pressure (90-180 mmHg): "))
heart_rate = int(input("Enter resting heart rate (50-120 bpm): "))
exercise_hours = float(input("Enter weekly exercise hours (0-15): "))

custom_data = pd.DataFrame({
    "age": [age],
    "cholesterol": [cholesterol],
    "blood_pressure": [blood_pressure],
    "heart_rate": [heart_rate],
    "exercise_hours": [exercise_hours]
})

custom_prediction = model.predict(custom_data)
prediction_proba = model.predict_proba(custom_data)

if custom_prediction == 1:
    print(f"High risk of heart disease ⚠️ (Probability: {prediction_proba[0][1]:.3f})")
else:
    print(f"Low risk of heart disease ✅ (Probability: {prediction_proba[0][0]:.3f})")

# PLOTS

# 1. Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Heart Disease Prediction")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Feature Importance
plt.figure(figsize=(10,6))
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=True)

plt.barh(importance_df['feature'], importance_df['importance'], color='lightcoral', edgecolor='black')
plt.xlabel("Importance")
plt.title("Random Forest: Feature Importance for Heart Disease Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Correct vs Wrong Predictions Scatter Plot (using two most important features)
# Get top 2 features
top_features = importance_df.tail(2)['feature'].values
correct_idx = predictions == y_test
wrong_idx = predictions != y_test

plt.figure(figsize=(8,6))

# Correct predictions
plt.scatter(X_test.loc[correct_idx, top_features[0]], 
            X_test.loc[correct_idx, top_features[1]], 
            color="green", alpha=0.6, label=f"Correct ✅ ({correct_idx.sum()})")

# Wrong predictions
plt.scatter(X_test.loc[wrong_idx, top_features[0]], 
            X_test.loc[wrong_idx, top_features[1]], 
            color="red", alpha=0.6, label=f"Wrong ❌ ({wrong_idx.sum()})")

# Custom input
plt.scatter(custom_data[top_features[0]].values[0], custom_data[top_features[1]].values[0], 
           color="black", marker="X", s=200, label="Your Input ⭐")

plt.xlabel(top_features[0].replace('_', ' ').title())
plt.ylabel(top_features[1].replace('_', ' ').title())
plt.title("Heart Disease: Correct vs Wrong Predictions")
plt.legend()
plt.savefig("correct_vs_wrong.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Risk Factor Analysis
plt.figure(figsize=(15, 10))

# Age distribution by disease status
plt.subplot(2, 3, 1)
no_disease = copyData[copyData["heart_disease"] == 0]["age"]
disease = copyData[copyData["heart_disease"] == 1]["age"]
plt.hist(no_disease, alpha=0.7, label="No Disease", bins=15, color='green')
plt.hist(disease, alpha=0.7, label="Disease", bins=15, color='red')
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()
plt.title("Age Distribution")

# Cholesterol distribution
plt.subplot(2, 3, 2)
no_disease = copyData[copyData["heart_disease"] == 0]["cholesterol"]
disease = copyData[copyData["heart_disease"] == 1]["cholesterol"]
plt.hist(no_disease, alpha=0.7, label="No Disease", bins=15, color='green')
plt.hist(disease, alpha=0.7, label="Disease", bins=15, color='red')
plt.xlabel("Cholesterol")
plt.ylabel("Frequency")
plt.legend()
plt.title("Cholesterol Distribution")

# Blood Pressure distribution
plt.subplot(2, 3, 3)
no_disease = copyData[copyData["heart_disease"] == 0]["blood_pressure"]
disease = copyData[copyData["heart_disease"] == 1]["blood_pressure"]
plt.hist(no_disease, alpha=0.7, label="No Disease", bins=15, color='green')
plt.hist(disease, alpha=0.7, label="Disease", bins=15, color='red')
plt.xlabel("Blood Pressure")
plt.ylabel("Frequency")
plt.legend()
plt.title("Blood Pressure Distribution")

# Heart Rate distribution
plt.subplot(2, 3, 4)
no_disease = copyData[copyData["heart_disease"] == 0]["heart_rate"]
disease = copyData[copyData["heart_disease"] == 1]["heart_rate"]
plt.hist(no_disease, alpha=0.7, label="No Disease", bins=15, color='green')
plt.hist(disease, alpha=0.7, label="Disease", bins=15, color='red')
plt.xlabel("Heart Rate")
plt.ylabel("Frequency")
plt.legend()
plt.title("Heart Rate Distribution")

# Exercise Hours distribution
plt.subplot(2, 3, 5)
no_disease = copyData[copyData["heart_disease"] == 0]["exercise_hours"]
disease = copyData[copyData["heart_disease"] == 1]["exercise_hours"]
plt.hist(no_disease, alpha=0.7, label="No Disease", bins=15, color='green')
plt.hist(disease, alpha=0.7, label="Disease", bins=15, color='red')
plt.xlabel("Exercise Hours")
plt.ylabel("Frequency")
plt.legend()
plt.title("Exercise Hours Distribution")

# Custom input visualization
plt.subplot(2, 3, 6)
patient_data = [age, cholesterol, blood_pressure, heart_rate, exercise_hours]
feature_labels = ['Age', 'Cholesterol', 'BP', 'HR', 'Exercise']
colors = ['red' if custom_prediction[0] == 1 else 'green'] * 5
plt.bar(feature_labels, patient_data, color=colors, alpha=0.7, edgecolor='black')
plt.title("Your Patient Profile")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("risk_factor_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nPlots saved as PNG files in the current directory!")