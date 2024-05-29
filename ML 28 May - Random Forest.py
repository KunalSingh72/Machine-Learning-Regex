#!/usr/bin/env python
# coding: utf-8

# In[1]:


# RandomForest Model ===> Supervised ML Model 

# 1 R.F.  ===> `100 DECISIO TREE MODEL 

# Target data Categorical ===> R.F.Classifier() ====> 100 dt ===> 70 yes , 30 no ===>
# majority ==> final prediction (yes)

# Target data Numerical ====>  R.F.Regressor() ====> mean > value ===> final prediction 


# In[2]:


import numpy as np 
import pandas as pd 


# In[3]:


df = pd.read_csv("C:\\Users\\saurabh\\Desktop\\Newdat\\Social_Network_Ads.csv")


# In[4]:


df.head(2)


# In[5]:


from sklearn.preprocessing import LabelEncoder


# In[6]:


lb = LabelEncoder() 


# In[7]:


df['Gender'] = lb.fit_transform(df['Gender'])


# In[8]:


df.head()


# In[9]:


x = df.drop(columns = ['Purchased'] , axis = 1)
y = df['Purchased']


# In[13]:


from sklearn.model_selection import train_test_split 


# In[14]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)


# In[15]:


from sklearn.ensemble import RandomForestClassifier 


# In[16]:


rf = RandomForestClassifier() 


# In[17]:


rf.fit(x_train , y_train) 


# In[18]:


y_pred = rf.predict(x_test)


# In[19]:


from sklearn.metrics import accuracy_score 


# In[20]:


accuracy_score(y_test , y_pred)


# In[21]:


from sklearn.linear_model import LogisticRegression 


# In[22]:


lr = LogisticRegression() 


# In[23]:


lr.fit(x_train , y_train) 


# In[24]:


y_pred = lr.predict(x_test)


# In[25]:


accuracy_score(y_test , y_pred)


# In[26]:


from sklearn.tree import DecisionTreeClassifier 


# In[27]:


dt = DecisionTreeClassifier() 


# In[28]:


dt.fit(x_train , y_train) 


# In[29]:


y_pred = dt.predict(x_test)


# In[30]:


accuracy_score(y_test , y_pred)


# In[ ]:





# # RandomForestRegressor 

# In[31]:


df = pd.read_csv("C:\\Users\\saurabh\\Desktop\\Newdat\\tips.csv")


# In[32]:


df.head(2)


# In[33]:


from sklearn.preprocessing import LabelEncoder 


# In[34]:


lb = LabelEncoder() 


# In[35]:


df['sex'] = lb.fit_transform(df['sex'])
df['smoker'] = lb.fit_transform(df['smoker'])
df['day'] = lb.fit_transform(df['day'])
df['time'] = lb.fit_transform(df['time'])


# In[36]:


df.head(2) 


# In[37]:


x = df.drop(columns = ['total_bill'] , axis = 1)
y = df['total_bill']


# In[38]:


from sklearn.model_selection import train_test_split 


# In[41]:


from sklearn.linear_model import LinearRegression 


# In[42]:


lr= LinearRegression()


# In[43]:


lr.fit(x_train , y_train) 


# In[44]:


y_pred = lr.predict(x_test)


# In[45]:


from sklearn.metrics import r2_score 


# In[46]:


r2_score(y_test , y_pred)


# In[47]:


from sklearn.tree import DecisionTreeRegressor 


# In[48]:


dt = DecisionTreeRegressor() 


# In[49]:


dt.fit(x_train , y_train) 


# In[50]:


y_pred= dt.predict(x_test)


# In[51]:


r2_score(y_test , y_pred)


# In[52]:


from sklearn.ensemble import RandomForestClassifier 


# In[54]:


rf = RandomForestClassifier() 


# In[55]:


rf.fit(x_train , y_train) 


# In[56]:


y_pred = rf.predict(x_test)


# In[57]:


r2_score(y_test , y_pred)


# In[ ]:




