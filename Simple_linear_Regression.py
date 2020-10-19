#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[2]:


df = pd.read_csv(r'C:\Users\prach_sxw8up\Downloads\AttendanceMarksSA.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
              


# In[6]:


x = df["MSE"]
y = df["ESE"]

sns.scatterplot(x,y)


# In[8]:


endog = df["ESE"]
exog = sm.add_constant(df[["MSE"]])
print(exog)


# In[9]:


# fit and summarize OLS model
mod = sm.OLS(endog, exog)
results = mod.fit()
print (results.summary())


# In[10]:


def RSE(y_true, y_predicted):
    
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))
    
    rse = math.sqrt(RSS / (len(y_true) - 2))
    return rse


# In[11]:


rse = RSE(df['ESE'],results.predict())
print(rse)


# In[12]:


x1 = df["Attendance"]
y1 = df["ESE"]

sns.scatterplot(x1, y1)


# In[16]:


endog1= df["ESE"]
exog1 = sm.add_constant(df[["Attendance"]])
print(exog1)


# In[17]:


# fit and summarize OLS model
mod1 = sm.OLS(endog1, exog1)
results1 = mod1.fit()
print (results1.summary())

