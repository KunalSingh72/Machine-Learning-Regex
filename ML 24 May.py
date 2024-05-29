#!/usr/bin/env python
# coding: utf-8

# # Function Transformer 

# In[1]:


from sklearn.preprocessing import FunctionTransformer


# In[2]:


import numpy as np


# In[3]:


#example 1
X = np.array([[1,2],[3,4]])

log_transform = FunctionTransformer(np.log1p)


# In[4]:


X_transformed = log_transform.transform(X)


# In[6]:


X_transformed


# # types of function transformer in machine learning

# In[10]:


#example 2
X = np.array([[1,2],[3,4]])
def my_feature_engineering(X):
    return np.hstack((X,X**2))


# In[11]:


customer_transformer = FunctionTransformer(my_feature_engineering)


# In[12]:


X_transformed = customer_transformer.transform(X)


# In[13]:


X_transformed


# In[14]:


#example 3
X = np.array([[1,2],[3,4]])
def my_scaling(X):
    return X/ np.max(X)


# In[15]:


custom_transformer = FunctionTransformer(my_scaling)


# In[17]:


X_trandformed = custom_transformer.transform(X)


# In[18]:


X_trandformed


# In[19]:


#example 4
X = np.array([[1,2],[3,np.nan]])


# In[20]:


def my_cleaning(X):
    X[np.isnan(X)] = 0
    return X


# In[21]:


custom_transformer = FunctionTransformer(my_cleaning)


# In[22]:


X_trandformed = custom_transformer.transform(X)


# In[23]:


X_trandformed


# In[ ]:




