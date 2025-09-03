import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import random

# Generate synthetic dataset
random.seed(42)
np.random.seed(42)

data = {
    "feature1": [],
    "feature2": [],
    "feature3": [],
    "feature4": [],
    "target": []
}

# Generate 500+ data points with realistic relationships
for i in range(600):
    feature1 = np.random.uniform(0, 10)
    feature2 = np.random.uniform(0, 10)
    feature3 = np.random.uniform(0, 10)
    feature4 = np.random.uniform(0, 10)
    
    # Create a linear relationship with some multicollinearity and noise
    target = (3 * feature1 + 
              2 * feature2 + 
              1.5 * feature3 + 
              0.8 * feature4 + 
              0.5 * feature1 * feature2 +  # Interaction term
              np.random.normal(0, 2))  # Noise
    
    data["feature1"].append(feature1)
    data["feature2"].append(feature2)
    data["feature3"].append(feature3)
    data["feature4"].append(feature4)
    data["target"].append(target)

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["feature1", "feature2", "feature3", "feature4"]]
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Ridge regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge regression model
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Print metrics
print("=== Ridge Regression Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

# Print coefficients
feature_names = ["feature1", "feature2", "feature3", "feature4"]
print("\nRidge Regression Coefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter values for prediction:")
feature1_input = float(input("Enter feature1 value (0-10): "))
feature2_input = float(input("Enter feature2 value (0-10): "))
feature3_input = float(input("Enter feature3 value (0-10): "))
feature4_input = float(input("Enter feature4 value (0-10): "))

custom_data = np.array([[feature1_input, feature2_input, feature3_input, feature4_input]])
custom_data_scaled = scaler.transform(custom_data)

custom_prediction = model.predict(custom_data_scaled)
print(f"Predicted target value: {custom_prediction[0]:.2f}")

# PLOTS

# 1. Coefficients visualization
plt.figure(figsize=(8,5))
plt.bar(feature_names, model.coef_, color='skyblue', edgecolor='black')
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Ridge Regression: Feature Coefficients")
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.show()

# 2. Actual vs Predicted with tolerance
tolerance = 0.1 * np.abs(y_test.values)  # 10% tolerance
correct = np.abs(predictions - y_test.values) <= tolerance
wrong = ~correct

plt.figure(figsize=(8,8))
plt.scatter(y_test[correct], predictions[correct], color='green', 
           label=f"Correct ✅ ({correct.sum()})", alpha=0.7)
plt.scatter(y_test[wrong], predictions[wrong], color='red', 
           label=f"Wrong ❌ ({wrong.sum()})", alpha=0.7)

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', label='Perfect Prediction')

# Custom prediction point
plt.scatter(custom_prediction, custom_prediction, color='black', marker='X', s=200, label="Your Input ⭐")

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Ridge Regression: Actual vs Predicted Values")
plt.legend()
plt.show()

# 3. Residuals plot
residuals = y_test - predictions
plt.figure(figsize=(8,6))
plt.scatter(predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Ridge Regression: Residuals Plot")
plt.show()

# 4. Regularization effect visualization (comparison with different alpha values)
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
coefficients = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)
    coefficients.append(ridge_model.coef_)

coefficients = np.array(coefficients)

plt.figure(figsize=(10, 6))
for i, feature in enumerate(feature_names):
    plt.plot(alphas, coefficients[:, i], marker='o', label=feature)

plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression: Coefficient Shrinkage with Regularization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.show()

# 5. Feature relationship visualization
plt.figure(figsize=(12, 8))

for i, feature in enumerate(feature_names, 1):
    plt.subplot(2, 2, i)
    plt.scatter(X_test[feature], y_test, alpha=0.6, label="Actual", color='blue')
    plt.scatter(X_test[feature], predictions, alpha=0.6, label="Predicted", color='red')
    
    # Plot custom input
    if feature == "feature1":
        plt.scatter(feature1_input, custom_prediction, color='black', s=100, marker='X', label='Your Input')
    elif feature == "feature2":
        plt.scatter(feature2_input, custom_prediction, color='black', s=100, marker='X', label='Your Input')
    elif feature == "feature3":
        plt.scatter(feature3_input, custom_prediction, color='black', s=100, marker='X', label='Your Input')
    elif feature == "feature4":
        plt.scatter(feature4_input, custom_prediction, color='black', s=100, marker='X', label='Your Input')
    
    plt.xlabel(feature)
    plt.ylabel("Target")
    plt.title(f"{feature} vs Target")
    plt.legend()

plt.tight_layout()
plt.show()

# 6. Learning curve (training vs validation error)
from sklearn.model_selection import validation_curve

param_range = np.logspace(-3, 2, 10)
train_scores, validation_scores = validation_curve(
    Ridge(random_state=42), X_train_scaled, y_train, param_name='alpha', 
    param_range=param_range, cv=5, scoring='neg_mean_squared_error'
)

train_scores_mean = -train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)
validation_scores_std = validation_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.semilogx(param_range, train_scores_mean, 'o-', color='blue', label='Training Error')
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1, color='blue')

plt.semilogx(param_range, validation_scores_mean, 'o-', color='red', label='Validation Error')
plt.fill_between(param_range, validation_scores_mean - validation_scores_std,
                validation_scores_mean + validation_scores_std, alpha=0.1, color='red')

plt.axvline(x=1.0, color='green', linestyle='--', label='Current Alpha=1.0')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression: Validation Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()