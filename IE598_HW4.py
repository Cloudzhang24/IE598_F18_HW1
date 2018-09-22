# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:34:19 2018

@author: cloud
"""

## Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import matplotlib.pyplot as plot

## Read Housing Data 
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt',
                 header=None,sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

## Exploratory Data Analysis
summary = df.describe()
print(summary)
corMat = DataFrame(df.corr())
plot.pcolor(corMat)
plot.show()

##Split data into training and test sets 
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.8, random_state=42)

##Linear model 
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

##Ridge Model 
from sklearn.linear_model import Ridge
ridge1 = Ridge(alpha=1.0)
ridge1.fit(X_train, y_train)
y_train_pred = ridge1.predict(X_train)
y_test_pred = ridge1.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals at alpha =1.0')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
print('At alpha =1.0,MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('At alpha =1.0,R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

ridge2 = Ridge(alpha=3.0)
ridge2.fit(X_train, y_train)
y_train_pred = ridge2.predict(X_train)
y_test_pred = ridge2.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals at alpha =1.0')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
print('At alpha =2.0,MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('At alpha =2.0,R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

ridge3 = Ridge(alpha=6.0)
ridge3.fit(X_train, y_train)
y_train_pred = ridge3.predict(X_train)
y_test_pred = ridge3.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals at alpha =1.0')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
print('At alpha =3.0,MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('At alpha =3.0,R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
##At alpha =1.0 gives the best performing model 

from sklearn.linear_model import Lasso
lasso1 = Lasso(alpha=1.0)
lasso1.fit(X_train, y_train)
y_train_pred = lasso1.predict(X_train)
y_test_pred = lasso1.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
print('At alpha = 1.0, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('At alpha = 1.0, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

lasso2 = Lasso(alpha=0.5)
lasso2.fit(X_train, y_train)
y_train_pred = lasso2.predict(X_train)
y_test_pred = lasso2.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
print('At alpha = 0.5, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('At alpha = 0.5, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

lasso3 = Lasso(alpha=0.1)
lasso3.fit(X_train, y_train)
y_train_pred = lasso3.predict(X_train)
y_test_pred = lasso3.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
print('At alpha = 0.1, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('At alpha = 0.1, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

## alpha = 0.1 gives the best performing model 

from sklearn.linear_model import ElasticNet
elanet1 = ElasticNet(alpha=1.0, l1_ratio=0.5)
elanet1.fit(X_train, y_train)
y_train_pred = elanet1.predict(X_train)
y_test_pred = elanet1.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
print('At alpha = 1.0, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('At alpha = 1.0, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


elanet2 = ElasticNet(alpha=0.1, l1_ratio=0.5)
elanet2.fit(X_train, y_train)
y_train_pred = elanet2.predict(X_train)
y_test_pred = elanet2.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
print('At alpha = 0.1, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('At alpha = 0.1, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))



elanet3 = ElasticNet(alpha=0.05, l1_ratio=0.5)
elanet3.fit(X_train, y_train)
y_train_pred = elanet3.predict(X_train)
y_test_pred = elanet3.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
print('At alpha = 0.05, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('At alpha = 0.05, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

## alpha=0.05 gives the best performing model 

print("My name is Xianhao Zhang")
print("My NetID is: xzhan137")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
