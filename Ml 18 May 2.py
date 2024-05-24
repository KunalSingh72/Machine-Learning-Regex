#!/usr/bin/env python
# coding: utf-8

# ## EDA => Exploratory Data Analysis

# ## Parts of EDA

# In[1]:


# 1. Univeriate Analysis ==> Analysis on single Independent column
# 2. Biveriate Analysis ==> Analysis on two columns
# 3. Multiveriate Analysis ==> Analysis on more than 2 columns


# In[2]:


# Data types
# 1. Numerical Data => Continuous
# 2. Categorical Data => Discrete


# In[3]:


import pandas as pd
import numpy as pandas
import matplotlib.pyplot as plt #Visualization Library
import seaborn as sns # Matplotlib's updated version


# In[4]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\titanic.csv")


# In[5]:


df.head()


# # 1. Univariate Analysis

# In[6]:


df.columns


# In[7]:


sns.countplot(x = df['Survived'])


# In[8]:


df['Survived'].value_counts()


# In[9]:


df['Survived'].value_counts().plot(kind = 'bar')


# In[10]:


sns.countplot(x = df['Pclass'])


# In[11]:


# If we want to find out percentage then use piechart
df['Survived'].value_counts()


# In[12]:


df['Survived'].value_counts().plot(kind = 'pie', autopct = '%.2f')


# In[13]:


plt.hist(df['Age'])


# In[14]:


sns.distplot(df['Age'])


# In[15]:


sns.distplot(df['Age'], hist =False)


# ## 3.BoxPLot

# In[16]:


#For Find our outlier

# 1.Lower fence
# 2.25% Data
# 3.IQR (Inter Quartile Range)(75%-25%)
# 4.75% Data
# Upper fence


# In[17]:


sns.boxplot(x=df['Age'])


# In[23]:


df2= sns.boxplot(x=df['Fare'])


# In[24]:


tips = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\tips.csv")

tips


# ## Bivariate Analysis

# In[25]:


# 1. Scatterplot(Numerical col - Numerical col)


# In[26]:


sns.scatterplot(x = tips['total_bill'], #df['total_bill']
                y = tips['tip'])


# In[27]:


sns.scatterplot(x='total_bill', y='tip', data=tips, hue = tips['sex'])


# In[28]:


sns.scatterplot(x = 'total_bill',
                y = 'tip', data = tips,
                hue = tips['sex'], style=tips['smoker'])


# In[29]:


p = pd.crosstab(df['Pclass'], df['Survived'])
p


# In[30]:


sns.heatmap(p)


# In[31]:


((df.groupby('Pclass').mean()['Survived'])*100).plot(kind = 'bar')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




