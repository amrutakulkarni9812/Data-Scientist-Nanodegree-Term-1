# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:14:07 2019

@author: amruta.kulkarni
"""
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Import dataset
train_data = pd.read_csv("polynomial_regression.csv")

# Assign the data to predictor and outcome variables
X = train_data['Var_X'].values.reshape(-1,1) 
# .values creates an array of column values: 1 row 20 columns
# .reshape(-1,1) converts it to unknown rows and 1 column
Y = train_data['Var_Y']

# Create polynomial features
# Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree = 4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
# Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression()
poly_model.fit(X_poly, Y)

# Plot
X_plot=np.linspace(0,1,100).reshape(-1,1)
X_plot_poly=poly_feat.fit_transform(X_plot)
plt.plot(X,Y,"b.")
plt.plot(X_plot_poly,poly_model.predict(X_plot_poly),'-r')
plt.show()