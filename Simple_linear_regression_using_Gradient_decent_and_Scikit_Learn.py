#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv(r'C:\Users\prach_sxw8up\Downloads\AttendanceMarksSA.csv')
df.head()


# In[4]:


x = df['MSE']
y = df['ESE']
sns.scatterplot(x,y)


# In[5]:


beta0 = 0
beta1 = 0
alpha = 0.01
count = 1000
n = float(len(x))


# In[16]:


for i in range(count):
    ybar = beta1*x  + beta0
    beta1 = beta1 - (alpha/n)*sum(x*(ybar-y))
    beta0 = beta0 - (alpha/n)*sum(ybar-y)
    
print(beta0, beta1)


# In[17]:


ybar = beta1*x + beta0

plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(ybar), max(ybar)], color = 'red') #regression line
plt.show()


# In[21]:


import math
def RSE(y_true, y_predicted):
    
    y_true =np.array(y_true)
    y_predicted =np.array(y_predicted)
    RSS =np.sum(np.square(y_true - y_predicted))
    
    rse = math.sqrt(RSS / (len(y_true) - 2))
    return rse

rse = RSE(df['ESE'],ybar)
print(rse)


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[23]:


x = np.array(df['MSE']).reshape(-1,1)
y = np.array(df['ESE']).reshape(-1,1)


lr = LinearRegression()
lr.fit(x,y)

print(lr.coef_)
print(lr.intercept_)

yp = lr.predict(x)
rse = RSE(y,yp)

print(rse)

