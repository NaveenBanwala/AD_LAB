
# import pandas as pd

# # Load dataset
# data = pd.read_csv("StudentPerformanceFactors.csv")

# # Separate numeric and categorical columns
# numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
# categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# print("Numeric columns:")
# print(numeric_cols)

# print("\nCategorical columns:")
# print(categorical_cols)

import pandas as pd
import scipy.stats as stats

# 1. Load your dataset
# Replace 'your_dataset.csv' with your actual file name
df = pd.read_csv('StudentPerformanceFactors.csv') 

# Define your columns based on your output
target = 'Exam_Score'

numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                'Tutoring_Sessions', 'Physical_Activity']

categorical_cols = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
                    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
                    'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
                    'Parental_Education_Level', 'Distance_from_Home', 'Gender']

# Dictionary to store results
p_values = {}

# 2. Calculate P-values for Numeric Features (Pearson Correlation)
print("--- Numeric Features (Pearson Correlation) ---")
for col in numeric_cols:
    if col in df.columns:
        # Drop NaN values for accurate calculation
        clean_data = df[[col, target]].dropna()
        stat, p_val = stats.pearsonr(clean_data[col], clean_data[target])
        p_values[col] = p_val
        print(f"{col}: p-value = {p_val:.5f}")

# 3. Calculate P-values for Categorical Features (ANOVA)
print("\n--- Categorical Features (ANOVA) ---")
for col in categorical_cols:
    if col in df.columns:
        clean_data = df[[col, target]].dropna()
        # Group scores by category
        groups = [group[target].values for name, group in clean_data.groupby(col)]
        
        # Perform ANOVA (requires at least 2 groups)
        if len(groups) > 1:
            stat, p_val = stats.f_oneway(*groups)
            p_values[col] = p_val
            print(f"{col}: p-value = {p_val:.5f}")
        else:
            print(f"{col}: Not enough categories to test.")

# 4. Summary: Significant Features (p < 0.05)
print("\n--- Summary: Significant Features (p < 0.05) ---")
significant_features = [feature for feature, p in p_values.items() if p < 0.05]
# Sort by p-value (lowest is most significant)
significant_features.sort(key=lambda x: p_values[x])

df_results = pd.DataFrame(list(p_values.items()), columns=['Feature', 'P-Value'])
df_results['Significant'] = df_results['P-Value'] < 0.05
print(df_results.sort_values(by='P-Value'))