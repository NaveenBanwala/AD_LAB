import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
try:
    df = pd.read_csv('email_spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'email_spam.csv' not found.")
    exit()

df = df.dropna(subset=['text', 'type'])

# 2. Features (X) and Target (y)
X = df['text']
y = df['type']

# 3. Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression(class_weight='balanced', C=10.0, solver='liblinear')
model.fit(X_train_vec, y_train)

# 6. Predictions
y_pred = model.predict(X_test_vec)

# 7. PERFORMANCE ASSESSMENT
print(f"--- Accuracy Score ---")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print(f"--- Classification Report ---")
# zero_division=0 removes warnings if a class is completely missed
print(classification_report(y_test, y_pred, zero_division=0))

print(f"--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 8. Visualizing the Results
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Spam Classification Confusion Matrix (Tuned)')
plt.show()