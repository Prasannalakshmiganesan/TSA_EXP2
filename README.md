# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib)

2. Load the dataset

3. Calculate the linear trend values using least square method

4. Calculate the polynomial trend values using least square method

5. End the program

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

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Print Linear Regression Formula
intercept_linear = linear_model.intercept_[0]
coefficients_linear = linear_model.coef_[0]
print(f"Linear Regression Formula: BookID = {intercept_linear:.2f} + "
      f"{coefficients_linear[0]:.2f} * Publication Year + "
      f"{coefficients_linear[1]:.2f} * Authors Count")

# Plot Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, label='Actual Data', color='blue')
plt.plot(X[:, 0], y_pred_linear, color='red', label='Linear Prediction')
plt.xlabel('Publication Year')
plt.ylabel('BookID')
plt.title('Linear Trend Estimation')
plt.legend()
plt.show()
```
B- POLYNOMIAL TREND ESTIMATION
```python
# Polynomial Regression Model (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

# Print Polynomial Regression Formula
intercept_poly = poly_model.intercept_[0]
coefficients_poly = poly_model.coef_[0]
print(f"Polynomial Regression Formula: BookID = {intercept_poly:.2f} + "
      f"{coefficients_poly[1]:.2f} * Publication Year + "
      f"{coefficients_poly[2]:.2f} * Authors Count + "
      f"{coefficients_poly[3]:.2f} * (Publication Year)^2 + "
      f"{coefficients_poly[4]:.2f} * (Publication Year * Authors Count) + "
      f"{coefficients_poly[5]:.2f} * (Authors Count)^2")

# Plot Polynomial Regression
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, label='Actual Data', color='blue')
plt.plot(X[:, 0], y_pred_poly, color='green', label='Polynomial Prediction (degree=2)')
plt.xlabel('Publication Year')
plt.ylabel('BookID')
plt.title('Polynomial Trend Estimation')
plt.legend()
plt.show()
```
### OUTPUT

A - LINEAR TREND ESTIMATION

![image](https://github.com/user-attachments/assets/5c6a11af-7ed3-4f83-b765-201815fbd5b2)


B- POLYNOMIAL TREND ESTIMATION

![image](https://github.com/user-attachments/assets/b211966f-66fd-41bd-8e64-531d7b98f194)



### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
