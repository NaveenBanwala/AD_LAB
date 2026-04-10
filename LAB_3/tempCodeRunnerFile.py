
import pandas as pd

# Load dataset
data = pd.read_csv("StudentPerformanceFactors.csv")

# Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

print("Numeric columns:")
print(numeric_cols)

print("\nCategorical columns:")
print(categorical_cols)
