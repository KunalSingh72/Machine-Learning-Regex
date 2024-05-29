#!/usr/bin/env python
# coding: utf-8

# # Task : DecisionTreeRegressor on Insurance

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\insurance.csv")


# In[5]:


df.head()


# In[6]:


from sklearn.preprocessing import LabelEncoder


# In[7]:


lb = LabelEncoder()


# In[8]:


df['sex'] = lb.fit_transform(df['sex'])
df['smoker'] = lb.fit_transform(df['smoker'])
df['region'] = lb.fit_transform(df['region'])


# In[9]:


df.head()


# In[10]:


x = df.drop(columns = ['charges'], axis = 1)
y = df['charges']


# In[11]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


# In[12]:


from sklearn.tree import DecisionTreeRegressor


# In[13]:


dt =DecisionTreeRegressor()
dt.fit(x_train, y_train)


# In[14]:


DecisionTreeRegressor()

y_pred = dt.predict(x_test)


# In[15]:


from sklearn.metrics import r2_score

r2_score(y_test, y_pred)


# # Decision Tree classifier on New Placement Data

# In[7]:


import numpy as np
import pandas as pd


# In[8]:


df =pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\newplacementdata.csv")


# In[9]:


df.head()


# In[10]:


df.dropna()


# In[11]:


x = df.drop(columns = ['placed'])
y = df['placed']


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


# In[14]:


from sklearn.tree import DecisionTreeClassifier 


# In[15]:


dt = DecisionTreeClassifier()


# In[16]:


dt.fit(x_train, y_train)


# In[17]:


y_pred = dt.predict(x_test)


# In[18]:


from sklearn.metrics import accuracy_score


# In[19]:


accuracy_score(y_test, y_pred)


# In[ ]:




