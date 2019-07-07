# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:10:06 2019

@author: amruta.kulkarni
"""

# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('regularization.csv', header = None)
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

# TODO: Create the linear regression model with lasso regularization.
# Regularization makes the model simpler and avoids overfittimg
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)

# Regularization zeroes out coefficients of first and sixth predictor variable 
# i.e. these 2 predictors are removed from the model