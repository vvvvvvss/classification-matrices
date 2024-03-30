import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score

# Load your stock data (make sure it contains features and target variable)
# For example, let's say you have a CSV file named 'stock_data.csv' with columns 'feature1', 'feature2', ..., 'target'
stock_data = pd.read_csv('data.csv')

# Split the data into features and target variable
X = stock_data[['Low', 'High', 'Open']]  # Features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt( mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R Squared:", r2)

# You can also print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

