#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd


# In[23]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\newplacementdata.csv")


# In[24]:


df.head()


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


plt.figure(figsize = (15,5))
plt.subplot(121)
sns.distplot(df['cgpa'])
plt.subplot(122)
sns.distplot(df['placement_exam_marks'])
plt.show()


# In[27]:


sns.boxplot(x = df['placement_exam_marks'])


# In[29]:


# finding th IQR
percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)


# In[30]:


percentile25


# In[31]:


percentile75


# In[34]:


IQR = percentile75 -percentile25
IQR


# In[35]:


upper_limit = percentile75 + 1.5*IQR
upper_limit


# In[36]:


lower_limit = percentile25-1.5*IQR
lower_limit


# In[37]:


df[df['placement_exam_marks']>upper_limit]


# In[39]:


df[df['placement_exam_marks']<lower_limit]


# In[40]:


# Trimming ( Outliers Removing technique 1)


# In[42]:


new_df = df[df['placement_exam_marks']<upper_limit]
new_df


# In[43]:


# Comparision 


# In[46]:


plt.figure(figsize = (15,5))
plt.subplot(221)
sns.distplot(df['placement_exam_marks'])

plt.subplot(222)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(223)
sns.distplot(new_df['placement_exam_marks'])

plt.subplot(224)
sns.boxplot(new_df['placement_exam_marks'])
plt.show()


# In[47]:


#capping (Outlier removing technique 2)


# In[48]:


new_df_cap = df.copy()


# In[49]:


#min = 5, max = 15

#min 4,3, 1
#max = 40, 30, 50

#updated_min_value = 1
#updated_max_value = 50


# In[50]:


new_df_cap['placement_exam_marks'] = np.where(
    
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    
    np.where(
        new_df_cap['placement_exam_marks']<lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)


# In[51]:


new_df_cap


# In[52]:


# comparison


# In[53]:


plt.figure(figsize = (15,8))
plt.subplot(221)
sns.distplot(df['placement_exam_marks'])

plt.subplot(222)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(223)
sns.distplot(new_df_cap['placement_exam_marks'])

plt.subplot(224)
sns.boxplot(new_df_cap['placement_exam_marks'])
plt.show()


# In[ ]:




