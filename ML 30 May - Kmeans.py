#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


from sklearn.cluster import KMeans


# In[4]:


#Generate synthetic data


# In[6]:


np.random.seed(0)
n_samples = 200
n_clusters = 4
x = np.random.rand(n_samples, 2) * 10


# In[8]:


# create K-means clustering model
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(x)


# In[9]:


# get cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_


# In[12]:


#visualize the clusters
plt.figure(figsize=(8,6))
plt.scatter(x[:,0], x[:,1], c= labels, cmap='viridis', edgecolor='k')
plt.scatter(centers[:,0], centers[:,1], c='red', marker='x',s=200)
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[14]:


import pandas as pd
df = pd.read_csv("E:\\Dataset\\mall - mall.csv")


# In[15]:


df.head()


# In[16]:


df = df.drop(columns = ['CustomerID', 'Genre'])


# In[17]:


df.head()


# In[18]:


x = df.iloc[:,[0,1]].values


# In[20]:


# x


# In[21]:


from sklearn.cluster import KMeans 


# In[22]:


import matplotlib.pyplot as plt


# In[25]:


a = []

for i in range(1,11):
    b = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    b.fit(x)
    a.append(b.inertia_)

plt.plot(range(1,11),a)

plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()


# # from the above plot, we can see the elbow point is at 4. So the number of clusters here will be 4                                                              

# In[29]:


b = KMeans(n_clusters = 4, init='k-means++', random_state = 90)
y_predict = b.fit_predict(x)


# In[28]:


plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s= 100, c= 'blue', label = 'Cluster 1')
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s= 100, c= 'green', label = 'Cluster 2')
plt.scatter(x[y_predict == 2, 0], x[y_predict == 2, 1], s= 100, c= 'red', label = 'Cluster 3')
plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s= 100, c= 'cyan', label = 'Cluster 1')

plt.scatter(b.cluster_centers_[:, 0], b.cluster_centers_[:, 1], s= 300, c='yellow', label = 'Centroid')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




