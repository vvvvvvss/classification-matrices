import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dengue prediction data (make sure it contains features and target variable)
# For example, let's say you have a CSV file named 'dengue_data.csv' with columns 'feature1', 'feature2', ..., 'target'
dengue_data = pd.read_csv('dataset.csv')

# Split the data into features and target variable
X = dengue_data[['Gender', 'Age','NS1', 'IgG', 'IgM', 'Area', 'AreaType', 'HouseType', 'District'  ]]  # Features
y = dengue_data['Outcome']                       # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assuming 'X_train' and 'X_test' contain your features
# Let's say 'gender' is a categorical variable in your dataset
# You need to convert it into numerical values using one-hot encoding
X_train_encoded = pd.get_dummies(X_train)  # Perform one-hot encoding on training data
X_test_encoded = pd.get_dummies(X_test)    # Perform one-hot encoding on testing data


# Initialize random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train_encoded, y_train)

# Make predictions
predictions = clf.predict(X_test_encoded)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, predictions))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
