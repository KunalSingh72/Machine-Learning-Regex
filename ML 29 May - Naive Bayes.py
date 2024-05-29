#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd


# In[19]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\Social_Network_Ads.csv")


# In[20]:


df.head()


# In[33]:


from sklearn.preprocessing import LabelEncoder


# In[34]:


lb = LabelEncoder()


# In[35]:


df['Gender'] = lb.fit_transform(df['Gender'])


# In[36]:


x = df.drop(columns = ['Purchased']) # Independent columns
y = df['Purchased'] #dependent column


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 23)


# In[39]:


from sklearn.preprocessing import StandardScaler


# In[40]:


sc = StandardScaler()


# In[41]:


x_train_new = sc.fit_transform(x_train)


# In[42]:


x_test_new = sc.transform(x_test)


# In[47]:


# from sklearn.naive_bayes import BurnolliNB
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

classifier = GaussianNB()


# In[48]:


classifier.fit(x_train_new, y_train)


# In[49]:


y_pred = classifier.predict(x_test_new)


# In[50]:


y_pred


# In[51]:


from sklearn.metrics import confusion_matrix


# In[52]:


cn = confusion_matrix(y_test, y_pred)


# In[53]:


cn


# In[57]:


# [[tp, fn]
#  [fp, tn]]


# In[58]:


from sklearn.metrics import accuracy_score


# In[59]:


accuracy_score(y_test, y_pred)


# In[ ]:




