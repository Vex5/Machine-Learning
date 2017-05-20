# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:36:43 2017

@author: Vedo
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Points_Mathematics_Data.csv')
#X is a matrix of independent variables
X = dataset.iloc[:, :-1].values
#Y is a vector of dependent variables
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Example of predicting the test set results
y_pred = regressor.predict(X_test)	

#Visualising the Data set
plt.scatter(X, y, color = 'red')
plt.title('Points achieved vs Learning Time')
plt.xlabel('Learning Time')
plt.ylabel('Points achieved')
plt.show()

#Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train)	, color = 'blue')
plt.title('Points achieved vs Learning Time (Training set)')
plt.xlabel('Learning Time')
plt.ylabel('Points achieved')
plt.show()

#Visualising the predicted Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Points achieved vs Learning Time (Test set)')
plt.xlabel('Learning Time')
plt.ylabel('Points achieved')
plt.show()


