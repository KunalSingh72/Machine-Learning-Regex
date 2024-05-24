#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Encoding ===> this is thje method to convert our categorical dayta inton numerical data.


# In[2]:


# (1). LabelEncoding ==> Using this method, we can convert our target or one dimansional data.


# In[3]:


import numpy as np
import pandas as pd


# In[4]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\covid_toy.csv")


# In[5]:


df.head(2)


# In[6]:


df=df.dropna()


# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


lb = LabelEncoder()


# In[9]:


df['gender'] = lb.fit_transform(df['gender'])
df['cough'] = lb.fit_transform(df['cough'])
df['city'] = lb.fit_transform(df['city'])
df['has_covid'] = lb.fit_transform(df['has_covid'])


# In[10]:


df.sample(5)


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


sc = StandardScaler()


# In[13]:


df_sc = sc.fit_transform(df)


# In[14]:


df_new = pd.DataFrame(df_sc , columns = df.columns)


# In[15]:


np.round(df.describe(),1)


# In[16]:


np.round(df_new.describe(),1)


# In[17]:


x = df.drop(columns = ['has_covid'], axis = 1)
y = df['has_covid']


# In[18]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)


# In[21]:


print(df.shape)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)


# In[22]:


from sklearn.preprocessing import MinMaxScaler


# In[23]:


mn = MinMaxScaler()


# In[24]:


x_train_mn = mn.fit_transform(x_train)


# In[25]:


x_test_mn = mn.fit_transform(x_test)


# In[26]:


x_train_new = pd.DataFrame(x_train_mn, columns = x_train.columns)


# In[27]:


np.round(x_train_new.describe(), 1)


# ## (2). Ordinal Encoder

# In[28]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\covid_toy.csv")


# In[ ]:





# In[30]:


df['city'].value_counts()


# In[31]:


df['cough'].value_counts()


# In[46]:


from sklearn.preprocessing import OrdinalEncoder


# In[47]:


df = df.drop(['age','fever'],axis=1)


# In[48]:


df['city'].value_counts()


# In[49]:


df['cough'].value_counts()


# In[53]:


oe = OrdinalEncoder(categories=[['Male', 'Female'],['Mild','Strong'], ['Kolkata', 'Bangalore', 'Delhi', 'Mumbai'], ['Yes','No']])


# In[54]:


oe 


# In[55]:


oe.fit(df)


# In[56]:


df_new = oe.transform(df)


# In[57]:


oe.categories_


# In[ ]:





# In[ ]:




