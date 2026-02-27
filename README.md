# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.

2.Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.

3.Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.

4.Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.1.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: S LOGESHWAR
RegisterNumber: 25014493 
*/

# Import required libraries
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample dataset
# Features: [House Size (sq.ft), Number of Rooms]
X = np.array([
    [800, 2],
    [1000, 3],
    [1200, 3],
    [1500, 4],
    [1800, 4],
    [2000, 5]
])

# Targets:
# y_price -> House Price (in lakhs)
# y_occupants -> Number of occupants
y_price = np.array([40, 50, 60, 75, 90, 110])
y_occupants = np.array([2, 3, 4, 5, 6, 7])

# Split the dataset
X_train, X_test, y_price_train, y_price_test = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

_, _, y_occ_train, y_occ_test = train_test_split(
    X, y_occupants, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create SGD Regressor models
price_model = SGDRegressor(max_iter=1000, learning_rate='optimal')
occupant_model = SGDRegressor(max_iter=1000, learning_rate='optimal')

# Train the models
price_model.fit(X_train_scaled, y_price_train)
occupant_model.fit(X_train_scaled, y_occ_train)

# Prediction
price_prediction = price_model.predict(X_test_scaled)
occupant_prediction = occupant_model.predict(X_test_scaled)

# Display results
print("Predicted House Prices:", price_prediction)
print("Actual House Prices:", y_price_test)

print("\nPredicted Number of Occupants:", occupant_prediction)
print("Actual Number of Occupants:", y_occ_test)
```

## Output:
<img width="798" height="160" alt="image" src="https://github.com/user-attachments/assets/cc2b799e-4097-4195-9be2-b116985a7720" />




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
