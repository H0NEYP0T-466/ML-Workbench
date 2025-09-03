import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# Load data
data = pd.read_csv("spam_detection_dataset.csv")

copyData = data.copy()

# Features and target
X = copyData[["word_frequency_urgent", "word_frequency_free", "word_frequency_money", 
              "capital_letters_ratio", "exclamation_marks", "email_length"]]
copyData["spam"] = copyData["spam"].map({"No": 0, "Yes": 1})
y = copyData["spam"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print metrics
print("=== Spam Detection Results ===")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Get user input for custom prediction
print("\n=== Custom Prediction ===")
print("Enter email characteristics for spam detection:")
urgent_freq = float(input("Enter frequency of word 'urgent' (0-10): "))
free_freq = float(input("Enter frequency of word 'free' (0-15): "))
money_freq = float(input("Enter frequency of word 'money' (0-12): "))
capital_ratio = float(input("Enter capital letters ratio (0-1): "))
exclamation_marks = int(input("Enter number of exclamation marks (0-20): "))
email_length = int(input("Enter email length in words (50-400): "))

custom_data = pd.DataFrame({
    "word_frequency_urgent": [urgent_freq],
    "word_frequency_free": [free_freq],
    "word_frequency_money": [money_freq],
    "capital_letters_ratio": [capital_ratio],
    "exclamation_marks": [exclamation_marks],
    "email_length": [email_length]
})

custom_prediction = model.predict(custom_data)
prediction_proba = model.predict_proba(custom_data)

if custom_prediction == 1:
    print(f"This email is likely SPAM 🚨 (Probability: {prediction_proba[0][1]:.3f})")
else:
    print(f"This email is likely HAM ✅ (Probability: {prediction_proba[0][0]:.3f})")

# PLOTS

# 1. Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Spam Detection")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Correct vs Wrong Predictions (using two most discriminative features)
correct_idx = predictions == y_test
wrong_idx = predictions != y_test

plt.figure(figsize=(8,6))

# Correct predictions
plt.scatter(X_test.loc[correct_idx, "word_frequency_free"], 
            X_test.loc[correct_idx, "exclamation_marks"], 
            color="green", alpha=0.6, label=f"Correct ✅ ({correct_idx.sum()})")

# Wrong predictions
plt.scatter(X_test.loc[wrong_idx, "word_frequency_free"], 
            X_test.loc[wrong_idx, "exclamation_marks"], 
            color="red", alpha=0.6, label=f"Wrong ❌ ({wrong_idx.sum()})")

# Custom input
plt.scatter(free_freq, exclamation_marks, color="black", marker="X", s=200, label="Your Input ⭐")

plt.xlabel("Word Frequency: 'Free'")
plt.ylabel("Exclamation Marks Count")
plt.title("Spam Detection: Correct vs Wrong Predictions")
plt.legend()
plt.savefig("correct_vs_wrong.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Feature Distribution Analysis
feature_names = ["word_frequency_urgent", "word_frequency_free", "word_frequency_money", 
                "capital_letters_ratio", "exclamation_marks", "email_length"]

plt.figure(figsize=(15, 10))

for i, feature in enumerate(feature_names, 1):
    plt.subplot(2, 3, i)
    
    # Split by spam status
    ham_data = copyData[copyData["spam"] == 0][feature]
    spam_data = copyData[copyData["spam"] == 1][feature]
    
    plt.hist(ham_data, alpha=0.7, label="Ham", bins=20, color='green', density=True)
    plt.hist(spam_data, alpha=0.7, label="Spam", bins=20, color='red', density=True)
    
    # Mark user input
    if feature == "word_frequency_urgent":
        plt.axvline(urgent_freq, color='black', linestyle='--', linewidth=2, label='Your Input')
    elif feature == "word_frequency_free":
        plt.axvline(free_freq, color='black', linestyle='--', linewidth=2, label='Your Input')
    elif feature == "word_frequency_money":
        plt.axvline(money_freq, color='black', linestyle='--', linewidth=2, label='Your Input')
    elif feature == "capital_letters_ratio":
        plt.axvline(capital_ratio, color='black', linestyle='--', linewidth=2, label='Your Input')
    elif feature == "exclamation_marks":
        plt.axvline(exclamation_marks, color='black', linestyle='--', linewidth=2, label='Your Input')
    elif feature == "email_length":
        plt.axvline(email_length, color='black', linestyle='--', linewidth=2, label='Your Input')
    
    plt.xlabel(feature.replace('_', ' ').title())
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"{feature.replace('_', ' ').title()} Distribution")

plt.suptitle("Feature Distributions: Ham vs Spam")
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Model Parameters Visualization
print("\n=== Naive Bayes Model Parameters ===")
feature_names_short = ["Urgent", "Free", "Money", "Capital", "Exclaim", "Length"]

# Create a comparison plot of means and variances
plt.figure(figsize=(12, 8))

# Means comparison
plt.subplot(2, 1, 1)
ham_means = model.theta_[0]  # Class 0 (Ham) means
spam_means = model.theta_[1]  # Class 1 (Spam) means

x = np.arange(len(feature_names_short))
width = 0.35

plt.bar(x - width/2, ham_means, width, label='Ham', color='green', alpha=0.7)
plt.bar(x + width/2, spam_means, width, label='Spam', color='red', alpha=0.7)

plt.xlabel('Features')
plt.ylabel('Mean Values')
plt.title('Feature Means by Class')
plt.xticks(x, feature_names_short)
plt.legend()

# Variances comparison
plt.subplot(2, 1, 2)
ham_vars = model.var_[0]  # Class 0 (Ham) variances
spam_vars = model.var_[1]  # Class 1 (Spam) variances

plt.bar(x - width/2, ham_vars, width, label='Ham', color='green', alpha=0.7)
plt.bar(x + width/2, spam_vars, width, label='Spam', color='red', alpha=0.7)

plt.xlabel('Features')
plt.ylabel('Variance Values')
plt.title('Feature Variances by Class')
plt.xticks(x, feature_names_short)
plt.legend()

plt.tight_layout()
plt.savefig("model_parameters.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. Prediction Probability Analysis
plt.figure(figsize=(10, 6))

# Get prediction probabilities for test set
test_probabilities = model.predict_proba(X_test)

# Plot probability distributions
plt.subplot(1, 2, 1)
ham_probs = test_probabilities[y_test == 0, 0]  # Ham emails, probability of being ham
spam_probs = test_probabilities[y_test == 1, 0]  # Spam emails, probability of being ham

plt.hist(ham_probs, alpha=0.7, label="Actual Ham", bins=20, color='green')
plt.hist(spam_probs, alpha=0.7, label="Actual Spam", bins=20, color='red')
plt.axvline(prediction_proba[0][0], color='black', linestyle='--', linewidth=2, label='Your Email')
plt.xlabel("Probability of Ham")
plt.ylabel("Frequency")
plt.title("Ham Probability Distribution")
plt.legend()

plt.subplot(1, 2, 2)
ham_probs_spam = test_probabilities[y_test == 0, 1]  # Ham emails, probability of being spam
spam_probs_spam = test_probabilities[y_test == 1, 1]  # Spam emails, probability of being spam

plt.hist(ham_probs_spam, alpha=0.7, label="Actual Ham", bins=20, color='green')
plt.hist(spam_probs_spam, alpha=0.7, label="Actual Spam", bins=20, color='red')
plt.axvline(prediction_proba[0][1], color='black', linestyle='--', linewidth=2, label='Your Email')
plt.xlabel("Probability of Spam")
plt.ylabel("Frequency")
plt.title("Spam Probability Distribution")
plt.legend()

plt.tight_layout()
plt.savefig("probability_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nPlots saved as PNG files in the current directory!")