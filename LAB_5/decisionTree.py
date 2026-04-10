import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and Train the Decision Tree Model
# We use 'entropy' or 'gini' to measure the quality of the split
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 4. Make Predictions
y_pred = clf.fit(X_train, y_train).predict(X_test)

# 5. Evaluate the Model
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# 6. Visualize the Tree
plt.figure(figsize=(15,10))

# Adding your Name and Roll No. 
# (x=0.5 is center, y=0.02 is near the bottom)
plt.figtext(0.5, 0.02, "Name: Naveen Banwala | Roll No: 23052251", 
            ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

plot_tree(clf, 
          filled=True, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names,
          rounded=True)

plt.title("Decision Tree Visualization (Iris Dataset)")
plt.show()