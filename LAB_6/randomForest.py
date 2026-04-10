import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Load data
df = pd.read_csv('titanic.csv')
df.rename(columns={'2urvived': 'Survived', 'sibsp': 'SibSp'}, inplace=True)
df = df.dropna(subset=['Survived'])

# 2. Advanced Preprocessing & Feature Engineering
# Fix Sex: Simple numeric mapping (0/1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Create FamilySize: Combining siblings/spouses and parents/children
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Handling missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('S') # Fill missing with most common port

# Feature Selection
features = ['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare']
X = df[features]
y = df['Survived']

# 3. Training/Testing Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Hyperparameter Tuning using GridSearch
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7, 10],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# 5. Model Evaluation
y_pred = best_rf.predict(X_test)

print(f"--- IMPROVED LAB RESULTS ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"1. Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n2. Classification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Died', 'Survived'])
disp.plot(cmap=plt.cm.Greens)
plt.title('Improved Confusion Matrix')
plt.show()
# # -------------------------------
# # 1. Load Dataset


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# import warnings

# warnings.filterwarnings('ignore')

# # 1. Load data
# # Note: Using your titanic.csv but treating it as the classification task
# df = pd.read_csv('titanic.csv')

# # 2. Preprocessing & Data Cleaning
# df.rename(columns={'2urvived': 'Survived', 'sibsp': 'SibSp'}, inplace=True)
# df = df.dropna(subset=['Survived'])

# # Feature selection
# features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
# X = df[features].copy()
# y = df['Survived']

# # Handling categorical data and missing values
# X['Sex'] = pd.to_numeric(X['Sex'], errors='coerce').fillna(0)
# X['Age'] = X['Age'].fillna(X['Age'].median())
# X['Fare'] = X['Fare'].fillna(X['Fare'].median())

# # 3. Training/Testing Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Implement Random Forest (Ensemble Method)
# rf_model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
# rf_model.fit(X_train, y_train)

# # 5. Model Evaluation
# y_pred = rf_model.predict(X_test)

# # --- Objective: Overall Accuracy ---
# accuracy = accuracy_score(y_test, y_pred)
# print(f"--- LAB RESULTS ---")
# print(f"1. Overall Accuracy: {accuracy:.4f}")

# # --- Objective: Classification Report ---
# print("\n2. Classification Report:")
# print(classification_report(y_test, y_pred))

# # --- Objective: Confusion Matrix ---
# cm = confusion_matrix(y_test, y_pred)
# print("\n3. Confusion Matrix (Raw Values):")
# print(cm)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Died', 'Survived'])
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix: Random Forest')
# plt.show()