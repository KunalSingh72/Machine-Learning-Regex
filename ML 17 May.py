#!/usr/bin/env python
# coding: utf-8

# # Pandas ==> It is an open source library that is used for handle data manipulations.`

# In[1]:


# Data structure : 
# (1). Series (2). Data Frames


# In[2]:


import pandas as pd


# In[3]:


a = pd.Series([1,24,25,36])
a


# In[4]:


type(a)


# ## (2). Dataframes

# In[5]:


data = {
    "Emp_Id" : [1,2,3,4,5,6,7,8],
    "Name" : ['Sam', 'Gaurav', 'Aniket', 'Raj', 'Deepak', 'Kunal', 'Mohit', 'Eren'],
    "Department" : ['IT', 'HR', 'HR', 'IT', 'Operations', 'IT', 'Operations', 'IT'],
    "Working_Hour" : [8,9,7,8,8,7,9,8]
}


# In[6]:


type(data)


# In[7]:


df = pd.DataFrame(data)
df


# In[8]:


df.head() #5 by default


# In[9]:


df.head(3)


# In[10]:


df.tail()


# In[11]:


df.tail(3)


# In[12]:


df.sample(4) # It returns random-indexed rows.


# In[13]:


df.describe() # It returns statically view of data.


# In[14]:


df.info() # Complete overview of data


# In[15]:


df.to_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\Emp_Info.csv")
df.to_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\new_Emp_Info.csv", index = False)


# In[16]:


df.to_excel("C:\\Users\\Acer\\Documents\\REGEX\\ML\\new2_Emp_Info.xlsx")


# ## How to read csv file into jupyter notebook 

# In[17]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\Emp_Info.csv")


# In[18]:


df.head()


# In[19]:


df['Department'].value_counts()


# In[20]:


data = {
    "Emp_Id" : [1,2,3,4,5,6,7,8],
    "Name" : ['Sam', 'Gaurav', 'Aniket', 'Raj', 'Deepak', 'Kunal', 'Mohit', 'Eren'],
    "Department" : ['IT', 'HR', 'HR', 'IT', 'Operations', 'IT', 'Operations', 'IT'],
    "Working_Hour" : [8,9,7,8,8,7,9,8]
}


# In[21]:


df = pd.DataFrame(data)
df


# In[22]:


df['Name'][0] = 'Kriti'


# In[23]:


df.head()


# In[24]:


df['Working_Hour'][3] = 7
df.head()


# In[25]:


df.loc[2:5, ["Name","Department"]]


# In[26]:


df.iloc[2:5, [1,2]]


# In[27]:


df.iloc[2:5, 1:4]


# In[28]:


df['Name'][0] = None


# In[29]:


df['Department'][3] = None
df['Department'][4] = None


# In[30]:


df


# In[31]:


df.isnull () # It will returns true if you have missing values otherwise return False


# In[32]:


df.isnull().sum()  # We can check Total missing values in our dataframe


# In[33]:


df = df.dropna() #remove rows containing none values
df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




