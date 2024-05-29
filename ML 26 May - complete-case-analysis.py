#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


df = pd.read_csv("C:\\Users\\saurabh\\Desktop\\pyth\\dsjob.csv")


# In[3]:


df.head() 


# In[4]:


df.isnull().mean()*100


# In[5]:


cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]
cols


# In[6]:


df[cols].sample(5)


# In[7]:


df['education_level'].value_counts()


# In[8]:


len(df[cols].dropna()) / len(df)


# In[9]:


new_df = df[cols].dropna()
df.shape, new_df.shape


# In[10]:


import matplotlib.pyplot as plt 


# In[15]:


fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['experience'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['experience'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)


# In[12]:


temp = pd.concat([
            # percentage of observations per category, original data
            df['enrolled_university'].value_counts() / len(df),

            # percentage of observations per category, cca data
            new_df['enrolled_university'].value_counts() / len(new_df)
        ],
        axis=1)

# add column names
temp.columns = ['original', 'cca']

temp


# In[13]:


temp = pd.concat([
            # percentage of observations per category, original data
            df['education_level'].value_counts() / len(df),

            # percentage of observations per category, cca data
            new_df['education_level'].value_counts() / len(new_df)
        ],
        axis=1)

# add column names
temp.columns = ['original', 'cca']

temp


# In[ ]:




