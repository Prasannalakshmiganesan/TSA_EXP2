# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:

IMPORTING PACKAGES
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
```

A - LINEAR TREND ESTIMATION
```python
data = pd.read_csv('Goodreadsbooks.csv', nrows=8)
data['publication_date'] = pd.to_datetime(data['publication_date'])
data['publication_year'] = data['publication_date'].dt.year
author_counts = data['authors'].value_counts()
author_mapping = author_counts.to_dict()
data['authors_count'] = data['authors'].map(author_mapping)

X = data[['publication_year', 'authors_count']].values
y = data[['bookID']].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(X[:, 0], y, label='Actual')
plt.plot(X[:, 0], y_pred, color='red', label='Linear Prediction')
plt.xlabel('Publication Year')
plt.ylabel('BookID')
plt.legend()
plt.show()
```
B- POLYNOMIAL TREND ESTIMATION
```python
poly = PolynomialFeatures(degree=2) 
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

plt.scatter(X[:, 0], y, label='Actual')
plt.plot(X[:, 0], y_pred, color='red', label='Polynomial Prediction')
plt.xlabel('Publication Year')
plt.ylabel('bookID')
plt.legend()
plt.show()
```
### OUTPUT
A - LINEAR TREND ESTIMATION
![image](https://github.com/user-attachments/assets/63e00113-4af6-4098-9d7f-04c730a506bc)

B- POLYNOMIAL TREND ESTIMATION
![image](https://github.com/user-attachments/assets/b4fe4eaa-048d-4d8c-aacc-7204436b258e)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
