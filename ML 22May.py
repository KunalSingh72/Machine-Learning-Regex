#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION TO MACHINE LEARNING

# In[39]:


# process ===> Data Preparation ==> ML Model ==> Performence Evaluation


# # 1) Data Preparation

# In[55]:


# Data ====>  Independent data (x) + Dependent Data (y)

# x===>x_train,x_test

# y==>y_train,y_test

#Data Prepare==> ml model ===> performance evaluation


# In[7]:


import numpy as np
import pandas as pd


# In[8]:


df=pd.read_csv("C:\\Users\\YASH\\OneDrive\\Desktop\\ty\\placement.csv")


# In[9]:


df


# In[11]:


df.shape


# In[12]:


x=df.drop(columns =['placed'],axis=1)  ### independent columns
y=df['placed'] ###Target column


# In[13]:


print(x.shape)
print(y.shape)


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[17]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Standardization ===> Data Mean = 0, Standard Deviation =0

# In[19]:


np.round(x_train.describe(),3)  #here 3 represensts the float value we want


# In[20]:


from sklearn.preprocessing import StandardScaler


# In[21]:


sc=StandardScaler()


# In[22]:


x_train_sc=sc.fit_transform(x_train)  ### fit means learn the parameters and transform means apply changes on data


# In[23]:


x_train_new=pd.DataFrame(x_train_sc,columns=x_train.columns)


# In[24]:


x_train_new.head(3)


# In[26]:


np.round(x_train_new.describe(),1)  ## looke below this the mean is 0 and SD is 1


# # Same implementation with insurance dataset

# # Standardization

# In[27]:


df=pd.read_csv("C:\\Users\\YASH\\OneDrive\\Desktop\\ty\\insurance.csv")


# In[28]:


df


# In[29]:


df.head()


# In[30]:


df=df.drop(columns=['sex','smoker','region'])


# In[31]:


df.head()


# In[33]:


df.shape


# In[34]:


x=df.drop(columns =['charges'],axis=1)  ### independent columns
y=df['charges'] ###Target column


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[37]:


print(df.shape)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)


# In[38]:


np.round(x_train_new.describe(),1)  ## looke below this the mean is 0 and SD is 1


# In[40]:


from sklearn.preprocessing import StandardScaler


# In[41]:


sc= StandardScaler()


# In[42]:


x_train_sc=sc.fit_transform(x_train)  ### fit means learn the parameters and transform means apply changes on data


# In[45]:


x_train_sc


# In[46]:


x_train_new=pd.DataFrame(x_train_sc,columns=x_train.columns)


# In[47]:


np.round(x_train_new.describe(),1)


# # Normalization ===> min=0,max=1

# In[49]:


from sklearn.preprocessing import MinMaxScaler


# In[50]:


mn=MinMaxScaler()


# In[51]:


x_train_mn=mn.fit_transform(x_train)


# In[52]:


x_train_new=pd.DataFrame(x_train_mn,columns=x_train.columns)


# In[53]:


np.round(x_train.describe(),1)


# In[ ]:


#tips data set standaradization and normalization

