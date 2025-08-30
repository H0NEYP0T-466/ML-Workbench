import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

model=LogisticRegression()
data = pd.read_csv("emails.csv")

copyData = data[['text', 'spam']].dropna()

copyData['spam'] = copyData['spam'].astype(str).str.strip()   
copyData = copyData[copyData['spam'].isin(['0', '1'])]        
copyData['spam'] = copyData['spam'].astype(int)              

X_text= copyData['text']
y = copyData['spam']



vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
prediction=model.predict(X_test)

print("Enter you custom mail?:")
custom_email = input()

custom_email_vectorized = vectorizer.transform([custom_email])
custom_email_prediction = model.predict(custom_email_vectorized)

if custom_email_prediction[0] == 1:
    print("The email is classified as Spam.")
else:
    print("The email is classified as Not Spam.")



pca = PCA(n_components=2, random_state=42)
X_2D = pca.fit_transform(X.toarray())
custom_2D = pca.transform(custom_email_vectorized.toarray())


print("Accuracy:", accuracy_score(y_test, prediction))
print("Precision:", precision_score(y_test, prediction))
print("Recall:", recall_score(y_test, prediction))
print("F1 Score:", f1_score(y_test, prediction))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, prediction))
print("\nClassification Report:\n", classification_report(y_test, prediction))

plt.figure(figsize=(10,6))


plt.scatter(X_2D[y==0, 0], X_2D[y==0, 1], c='green', alpha=0.5, label='Not Spam')


plt.scatter(X_2D[y==1, 0], X_2D[y==1, 1], c='red', alpha=0.5, label='Spam')


plt.scatter(custom_2D[0,0], custom_2D[0,1], c='blue', s=150, marker='*', label='Custom Email')

plt.title("Spam vs Not Spam Emails (2D PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x=copyData['spam'])
plt.title("Spam vs Not Spam Emails")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0,1], ['Not Spam', 'Spam'])
plt.show()

cm = confusion_matrix(y_test, prediction)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()