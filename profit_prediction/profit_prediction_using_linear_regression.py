#!/usr/bin/env python
# coding: utf-8

# In[149]:


#importing the libraries for predicting profit for companies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[150]:


#importing dataset
companies = pd.read_csv(r"C:\Users\G.SAI KRISHNA\Desktop\ML_Projects\profit_prediction\1000_Companies.csv")
#Independent var's
X = companies.iloc[:,:-1].values #fetch rows until second last column
#Dependent Var i.e. Profit
Y = companies.iloc[:,4].values

companies.head()


# In[151]:


#Data visualization through Correlation Matrix
sns.heatmap(companies.corr())


# In[152]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#Encode Country Column
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)

X


# In[153]:


X=X[:,1:]
print(X)


# In[154]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[155]:


#fitting multiple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


# In[156]:


Y_pred = regressor.predict(X_test)
print(Y_pred)


# In[157]:


print(regressor.coef_)


# In[158]:


print(regressor.intercept_)


# In[159]:


from sklearn.metrics import r2_score
r2_score(Y_test,Y_pred)

