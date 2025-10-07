import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris dataset from scikit-learn
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['species'] = iris.target

# Display first few rows
print(iris_data.head())

# Split the data into features {x} and labels {y}
x = iris_data.drop(columns=['species'])
y = iris_data['species']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train ML model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Predict using the trained model
#predictions = model.predict([[4.6, 3.5, 1.5, 0.2]])
predictions = model.predict(pd.DataFrame([[4.6, 3.5, 1.5, 0.2]], columns=x.columns))
print(f"Prediction: {predictions}")
