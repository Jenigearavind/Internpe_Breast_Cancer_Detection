import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
dataset = sklearn.datasets.load_breast_cancer()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
print("First 5 rows of the dataset:")
print(df.head())
print("\nDescriptive Statistics:")
print(df.describe())
plt.figure(figsize=(6, 4))
df['target'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Distribution of Malignant vs Benign Tumors')
plt.xlabel('Target (0: Malignant, 1: Benign)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("\nAccuracy of Logistic Regression Model:")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
user_input = input("\nDo you want to change the test size?: ")
if user_input:
    try:
        test_size = float(user_input)
        if 0 < test_size < 1:
            print(f"\nAdjusting test size to {test_size}...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nNew Accuracy of Logistic Regression Model with test size {test_size}: {accuracy}")
            cm = confusion_matrix(y_test, y_pred)
            print("\nNew Confusion Matrix:")
            print(cm)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
            plt.title('New Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
        else:
            print("\nInvalid test size. Please enter a value between 0 and 1.")
    except ValueError:
        print("\nInvalid input. Test size must be a numeric value between 0 and 1.")
else:
    print("\nUsing the default test size of 0.3.")
