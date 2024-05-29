#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Decision Tree Algorithm ==> Supervised Ml algo

# Target data : 
# Categorical ==> DecisionTreeClassifier
# Numerical ==> DecisionTreeRegressor


# # DecisionTreeClassifier

# In[4]:


import numpy as np
import pandas as pd


# In[6]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\covid_toy.csv")


# In[7]:


df.head()


# In[8]:


df =df.dropna()


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


lb = LabelEncoder()


# In[11]:


df['gender'] = lb.fit_transform(df['gender'])
df['cough'] = lb.fit_transform(df['cough'])
df['city'] = lb.fit_transform(df['city'])
df['has_covid'] = lb.fit_transform(df['has_covid'])


# In[12]:


df.head()


# In[17]:


x = df.drop(columns = ['has_covid'])
y = df['has_covid']


# In[18]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


# In[22]:


from sklearn.tree import DecisionTreeClassifier 


# In[23]:


dt = DecisionTreeClassifier()


# In[24]:


dt.fit(x_train, y_train)


# In[25]:


y_pred = dt.predict(x_test)


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


accuracy_score(y_test, y_pred)


# # DecisionTreeRegressor

# In[28]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\tips.csv")


# In[29]:


df.head()


# In[30]:


from sklearn.preprocessing import LabelEncoder


# In[32]:


df['sex'] = lb.fit_transform(df['sex'])
df['smoker'] = lb.fit_transform(df['smoker'])
df['day'] = lb.fit_transform(df['day'])
df['time'] = lb.fit_transform(df['time'])


# In[33]:


df.head()


# In[35]:


x = df.drop(columns = ['total_bill'], axis = 1)
y = df['total_bill']


# In[36]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


# In[37]:


from sklearn.tree import DecisionTreeRegressor


# In[38]:


dt =DecisionTreeRegressor()
dt.fit(x_train, y_train)


# In[39]:


DecisionTreeRegressor()

y_pred = dt.predict(x_test)


# In[40]:


from sklearn.metrics import r2_score

r2_score(y_test, y_pred)


# In[ ]:




