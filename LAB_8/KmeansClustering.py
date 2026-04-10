import os
# FIX: Prevents the joblib/wmic error on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATASET
# Ensure 'Mall_Customers.csv' is in the same folder as this script
try:
    df = pd.read_csv('Mall_Customers.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Mall_Customers.csv not found. Please check the file path.")
    exit()

# 2. SELECT FEATURES
# X1: 2D Clustering (Income vs Spending)
# X2: 3D Clustering (Age, Income, Spending)
X1 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
X2 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# 3. FEATURE SCALING
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
X2_scaled = scaler.fit_transform(X2)

# 4. FINDING OPTIMAL K (ELBOW METHOD)
wcss = []
for i in range(1, 11):
    # n_init='auto' is the updated standard for KMeans
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X1_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', color='purple')
plt.title('The Elbow Method (Finding Optimal K)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 5. K-MEANS CLUSTERING (K=5)
k = 5
model = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
y_kmeans = model.fit_predict(X1_scaled)

# Add the cluster labels back to our dataframe for analysis
df['Cluster'] = y_kmeans

# 6. 2D VISUALIZATION
plt.figure(figsize=(12, 7))
sns.scatterplot(
    x=X1_scaled[:, 0], y=X1_scaled[:, 1], 
    hue=y_kmeans, palette='viridis', s=100, edgecolor='black', legend='full'
)

# Plotting Centroids
centroids = model.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='X')

plt.title('Customer Segments: Income vs Spending')
plt.xlabel('Annual Income (Standardized)')
plt.ylabel('Spending Score (Standardized)')
plt.legend(title='Clusters')
plt.show()

# 7. 3D VISUALIZATION (Age, Income, Spending)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Re-run model for 3D data
model_3d = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
y_3d = model_3d.fit_predict(X2_scaled)

ax.scatter(X2_scaled[:, 0], X2_scaled[:, 1], X2_scaled[:, 2], c=y_3d, cmap='plasma', s=40)

ax.set_title('3D Customer Segmentation')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending Score')
plt.show()

# 8. CLUSTER SUMMARY
print("\n--- Cluster Analysis Summary ---")
numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
print(df.groupby('Cluster')[numeric_cols].mean())



# 9. MULTI-DIMENSIONAL VISUALIZATION (Pair Plot)
# We select the main numerical columns and the Cluster label
plot_df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']]

# Setting the aesthetic style
sns.set_theme(style="whitegrid")

# Create the Pair Plot
pair_plot = sns.pairplot(
    plot_df, 
    hue='Cluster', 
    palette='bright', 
    diag_kind='kde', # Shows the distribution of each feature per cluster
    height=3, 
    aspect=1.2
)

pair_plot.fig.suptitle('Multi-Dimensional Cluster Relationships', y=1.02, fontsize=16)
plt.show()