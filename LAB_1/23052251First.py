import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.ion() 

import seaborn as sns

from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler


iris = load_iris()

df =pd.DataFrame(iris.data, columns=iris.feature_names)

df['species'] = iris.target
print(df['species'])

df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print(df.head())



# 3. Handling Missing Values

df.iloc[5, 0] = np.nan
df.iloc[10, 2] = np.nan

print("\nMissing Values Before Handling:")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())


#Encoding of features
df['species_encoded'] = df['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})

print("\nEncoded Data:")
print(df[['species', 'species_encoded']].head())

#5. Feature Scaling (Standardization)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.iloc[:,0:4]) 
# print(scaled_features)
scaled_df = pd.DataFrame(scaled_features, columns=df.columns[0:4])
print("\nScaled_features ")
print(scaled_df.head())


#6. Plot Distribution using matplotlib.pyplot.hist()
plt.figure()
plt.hist(df["sepal length (cm)"], bins=20) #Binning 
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

#7. Scatter Plot using seaborn.scatterplot()
plt.figure()
sns.scatterplot(
    x=df['sepal length (cm)'],
    y=df['petal length (cm)'],
    hue=df['species']
)
plt.title("Sepal Length vs Petal Length")
plt.show()


# 8. Correlation Heatmap using seaborn.heatmap()
correlation_matrix = df.iloc[:,0:4].corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True,cmap ='coolwarm', linewidths =0.5)
plt.title("Correlation Heatmap of Features") 
plt.show(block=True)