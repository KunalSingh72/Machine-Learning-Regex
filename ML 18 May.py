#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\placement.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


top_left_corner_df = df.iloc[:5,:3]
top_left_corner_df


# In[8]:


s = df.axes #no. of rows
s


# In[9]:


b = df.empty #find the missing value it return true otherwise false
b


# In[37]:


i = df.ndim  #dimension
i


# In[11]:


t = df.shape #shape = n(rows)*n(columns)
t


# In[12]:


d = df.size #row count * column count
d


# In[13]:


a = df.values # get a numpy array for df
a


# In[14]:


df = df.copy()


# In[15]:


p = df.sort_values(by = 'resume_score')
p


# In[16]:


r = df.sort_index()
r


# In[17]:


# x = df.astype(int) #type conversion
x = df['cgpa'].astype(int)
x


# In[18]:


t = df.abs()
t


# In[19]:


t = df.add(4)
t


# In[20]:


s = df.count()
s


# In[21]:


p = df.max()
p


# In[22]:


q = df.min()
q


# In[23]:


df.mean()


# In[24]:


df.median()


# In[25]:


df.sum()


# In[26]:


df.filter(items = ['cgpa', 'placed'])

# df(['cgpa','placed'])


# In[39]:


df.filter(items = [5,6], axis = 0) 


# In[28]:


df.filter(like = '5', axis=0)


# In[29]:


dic = df.to_dict()
dic


# In[30]:


df.to_string()


# In[31]:


idx = df.columns
idx


# In[32]:


label = df.columns[0]
label


# In[33]:


df.columns.tolist()


# In[34]:


df.columns.values


# In[35]:


p = df.rename(columns = {'cgpa':'half_yearly_marks', 'resume_score': 'semester_marks'})
p


# In[36]:


df['half'] = df['cgpa'].where(df['cgpa']>50, other = 0)
df.head(10)


# In[ ]:




