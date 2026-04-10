import os
import pandas as pd

# Get the directory where the script itself is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Combine it with the filename
file_path = os.path.join(base_path, 'email_spam.csv')

# Load the dataset using the full path
df = pd.read_csv(file_path)