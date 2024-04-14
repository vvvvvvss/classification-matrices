import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dengue_data = pd.read_csv('dataset.csv')

X = dengue_data[['Gender', 'Age','NS1', 'IgG', 'IgM', 'Area', 'AreaType', 'HouseType', 'District'  ]]  # Features
y = dengue_data['Outcome']                      

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_encoded = pd.get_dummies(X_train)  
X_test_encoded = pd.get_dummies(X_test)    

clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train_encoded, y_train)

predictions = clf.predict(X_test_encoded)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
