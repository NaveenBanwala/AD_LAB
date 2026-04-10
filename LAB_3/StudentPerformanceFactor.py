import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Load the Dataset
# Ensure the file path matches your local directory
df = pd.read_csv('StudentPerformanceFactors.csv')

# 2. Feature Selection (Based on your P-value analysis)
# Dropping non-significant columns
cols_to_drop = ['Sleep_Hours', 'School_Type', 'Gender']
df = df.drop(columns=cols_to_drop)

# 3. Preprocessing: Handle Categorical Data
# Linear Regression requires numerical input, so we encode categories
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define Features (X) and Target (y)
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']

# 4. Split Data into Training and Testing Sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make Predictions
y_pred = model.predict(X_test)

# 7. Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Model Performance ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# 8. Visualization: Actual vs Predicted Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')

# Add a diagonal line for reference (Perfect Prediction)
# If points fall on this line, the prediction is perfect
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)

plt.xlabel('Actual Exam Score')
plt.ylabel('Predicted Exam Score')
plt.title('Actual vs Predicted Exam Scores')
plt.grid(True)
plt.show()

# Optional: View Coefficients to see impact of each feature
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\n--- Feature Coefficients ---")
print(coefficients.sort_values(by='Coefficient', ascending=False))