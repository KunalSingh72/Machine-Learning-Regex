#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv("C:\\Users\\Acer\\Documents\\REGEX\\ML\\Attrition.csv")
df.head()


# # Employee Attrition Analysis

# # Is a type of behavioural analysis where we study the behaviour and characteristics of the employee who left the organization and compare their characteristics with the current employee to find the employee who may leave the organization soon.

# # a high rate of employee can be expensive for any company in terms of recruitment and training costs, loss of productivity and morale reduction of employees. By indentifying the causes of attrition, a company 

# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"


# In[5]:


df.isnull().sum()


# In[6]:


attr_df = df[df['Attrition'] == 'Yes']


# In[7]:


attrition_by = attr_df.groupby(['Department']).size().reset_index(name = 'count')
attrition_by


# In[8]:


attrition_by = attr_df.groupby(['EducationField']).size().reset_index(name = 'Count')
attrition_by


# In[9]:


attrition_by = attr_df.groupby(['Department']).size().reset_index(name='Count')


# In[10]:


fig = go.Figure(data=[go.Pie(
    labels = attrition_by['Department'],
    values = attrition_by['Count'],
    hole=0.4,
    marker= dict(colors=['#3CAEA3', '#F6D55C']),
    textposition = 'inside'
)])


# In[11]:


fig.update_layout(title='Attrition by Department', font=dict(size=16), legend=dict(
    orientation='h', yanchor='bottom', y = 1.02, xanchor = "right", x=1
))
fig.show()


# In[12]:


attrition_by = attr_df.groupby(['EducationField']).size().reset_index(name = 'Count')
attrition_by


# In[13]:


fig = go.Figure(data=[go.Pie(
    labels = attrition_by['EducationField'],
    values = attrition_by['Count'],
    hole=0.4,
    marker= dict(colors=['#3CAEA3', '#F6D55C']),
    textposition = 'inside'
)])


# In[14]:


fig.update_layout(title='Attrition by Education Field', font=dict(size=16), legend=dict(
    orientation='h', yanchor='bottom', y = 1.02, xanchor = "right", x=1
))
fig.show()


# # we can see that the employees with life Sciences as an education field have high attrition rate. Now let's have a look at the percentage of attrition by number of years at the company

# In[15]:


attrition_by = attr_df.groupby(['YearsAtCompany']).size().reset_index(name='Count')
attrition_by


# In[16]:


attrition_by = attr_df.groupby(['YearsSinceLastPromotion']).size().reset_index(name='Count')
attrition_by


# # We can see that the employees who don't get promotions leave the organization more compared to the employees who got promotions. Now let's have a look at the percentage of attrition by gender

# In[17]:


attrition_by = attr_df.groupby(['Gender']).size().reset_index(name='Count')
attrition_by


# In[18]:


fig = go.Figure(data=[go.Pie(
    labels = attrition_by['Gender'],
    values = attrition_by['Count'],
    hole=0.4,
    marker= dict(colors=['#3CAEA3', '#F6D55C']),
    textposition = 'inside'
)])


# In[19]:


fig.update_layout(title='Attrition by Gender', font=dict(size=16), legend=dict(
    orientation='h', yanchor='bottom', y = 1.02, xanchor = "right", x=1
))
fig.show()


# # Men have a high attrition rate compared to women. Now let's have a look at the attrition by analyzing the relationship between montly income and the age of the employees:

# In[20]:


import plotly.express as px


# In[21]:


fig = px.scatter(df, x='Age', y='MonthlyIncome', color='Attrition', trendline='ols')


# In[22]:


fig.update_layout(title="Age vs Monthly Income by Attrition")


# In[ ]:




