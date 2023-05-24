#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction Model Using Linear Regression

# In[5]:


# in this model we will predict the price of 1500 (sqft) based on a Dataset 

# first let's Draw Linear Regression 

import matplotlib.pyplot as plt
from scipy import stats

x = [1700,2100,1900,1300,1600,2200]          # x Represent the size of the house in square feet
y = [53000,66000,59000,43000,45000,67000]    # y Represent the price of the house dollars

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.plot(x, mymodel, color = 'black')
plt.title('size vs price')
plt.xlabel('size (sqft)')
plt.ylabel('price in dollars')
plt.show()


# In[9]:


#antehr method to Draw Linear Regression using skleaern.linear_model 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1700,2100,1900,1300,1600,2200])
y = np.array([53000,66000,59000,43000,45000,67000])

linreg = LinearRegression()

x = x.reshape (-1,1)

linreg.fit(x,y)

y_pred = linreg.predict(x)

plt.scatter(x,y)
plt.plot(x, y_pred, color = 'black')
plt.title('size vs price')
plt.xlabel('size (sqft)')
plt.ylabel('price in dollars')
plt.show()


# In[10]:


# Let't find R

# It is important to know how the relationship between the values of the x-axis and the values of the y-axis is, if there are no relationship the linear regression can not be used to predict anything.

# This relationship - the coefficient of correlation - is called `r`

# Python and the Scipy module will compute this value for you, all you have to do is feed it with the x and y values.

import numpy
from scipy import stats

x = [1700,2100,1900,1300,1600,2200]
y = [53000,66000,59000,43000,45000,67000]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)


# In[8]:


# Let's Predict the price of 1500(sqft) house

from scipy import stats

x = [1700,2100,1900,1300,1600,2200]
y = [53000,66000,59000,43000,45000,67000]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

price = myfunc(1500)

print(price)


# In[ ]:




