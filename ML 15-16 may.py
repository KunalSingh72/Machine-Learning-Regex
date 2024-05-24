#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 

li = []
size = int(input("Enter size of Array : "))

for i in range(size):
    val = int(input(f"Enter value {i+1} : "))
    li.append(val)
arr = np.array(li)
print(arr)


# # How to check shape and size of an array?
# 

# In[2]:


print(arr.size, arr.shape)


# In[3]:


a = [[1,2,3], [4,5,6], [7,8,9]]
b = np.array(a)
b


# In[4]:


print("Total Shape = ",b.shape)
print("Total Element = ", b.size)


# In[5]:


'''
r1 = [1,2,3]
r2 = [4,5,6]
r3 = [7,8,9]

c1 = [1,4,7]
c2 = [2,5,8]
c3 = [3,6,9]
'''


# In[6]:


#Image ===> pixels ===> (0-255)px ===> 0px (complete black), 255px
#convert ===> grayscale Image ===>
# Image(Pixels) ===> Normalization(0-1) ===> 0px black, 1px white

# 0, 1 ===> Neoron System

#Matrix ===> rows, columns
#Symmetric Matrix ===> n(rows) = n(colemns)
#Asymmetric Matrix ===> n(rows) != n(columns)
#Diagonal Elements = [(1,1), (2,2), (3,3)]


# # (1). zeros ===> It will crate an array in which all the elements are zero.

# In[7]:


a= np.zeros(4)
a


# In[8]:


a = np.zeros((3,4))
a


# # (2). ones() ===> It will crate an array in which all the elements are one.

# In[9]:


a = np.ones(3)
a


# In[10]:


b = np.ones((2,3))
b


# # (3). eye() ===> This function will create an array with diagonal values as 1s and rest all are 0s

# In[11]:


a = np.eye(3,4) #Asymmetric
a


# In[12]:


a = np.eye(4) #Symmetric  Matrix
a


# # (4). Diag ==> this function creates a two dimensional array with all the diagonal element as the given value and rest are zeros

# In[13]:


import numpy as np
a = np.diag([1,4,8,9])
a


# # (5). randint ==> this function is used to generate a random number between a given range.
# 
# syntax == randint(min_value, max_value, totalnumber)

# In[14]:


import numpy as np
a = np.random.randint(1,10,3)
a


# # (6). This fuinction is used to generate a random number between 0 to 1.
# 
# Syntax  == rand(number of values)

# In[15]:


import numpy as np
a = np.random.rand(5)
a


# # (7). randn() ==> This function is used to generate a random a number from -3 to 3. This may return postive or negative number as well.
# 
# Syntax ==> random.randn(number of values)

# In[16]:


import numpy as np
a = np.random.randn(5)
a


# # Reshpaing data

# In[17]:


import numpy as np
a =np.random.randint(0,50,12)
a


# # n(rows)*n(columns) = n(total_elements)

# In[18]:


a.shape


# In[19]:


a = a.reshape(2,6)
a


# In[20]:


a = a.reshape(3,4)
a


# In[21]:


a = a.reshape(4,3)
a


# In[22]:


a = a.reshape(6,2)
a


# In[23]:


a = a.reshape(12,1)
a


# # Q : for 64 `

# In[24]:


import numpy as np
a =np.random.randint(0,100,64)
a


# In[25]:


a = a.reshape(1,64)
a


# In[26]:


a = a.reshape(2,32)
a


# In[27]:


a = a.reshape(4,16)
a


# In[28]:


a = a.reshape(8,8)
a


# In[29]:


a = a.reshape(16,4)
a


# In[30]:


a = a.reshape(32,2)
a


# In[31]:


a = a.reshape(64,1)
a


# # Principle of -1

# In[33]:


a.reshape(-1,8)


# # seed function() ==> We know that randint function generates random numbers. Everytime we run the program, new set of random number is genrated. So, solve this problem we will use seed function.
# 

# In[74]:


import numpy as np
a= np.random.randint(1,100,10)
np.random.seed(1)
a


# ## View vs Copy ==> When we slice a sub-array from an array, It may be done by two ways.
# 

# ## View

# In[75]:


import numpy as np
a = np.array([12,12,53,23,24,13,4,32])
b = a[3:5]
b[:] = 0
a


# ## Copy

# In[76]:


a = np.array([12,12,53,23,24,13,4,32])
b = a[3:5].copy()
b[:] = 0
a


# ## conditional selection

# In[77]:


import numpy as np
a = np.arange(1,16)
a


# In[78]:


a>10


# In[79]:


a<10


# In[80]:


b = a>10
a[b]


# In[81]:


a[a%2==0]


# ## Operations on Array

# In[82]:


import numpy as np
a = np.arange(1,5)
a*2


# In[83]:


a+2


# In[84]:


a**2


# In[85]:


a = np.array([1,2,3,4]).reshape(2,2)
a


# In[86]:


b = np.array([5,6,7,8]).reshape(2,2)
b


# In[87]:


a+b


# In[88]:


b-a


# In[89]:


b/a


# In[90]:


b*a


# In[91]:


a.dot(b)


# In[92]:


import numpy as np
a = np.array([10,20,30,40,50])
np.min(a)


# In[93]:


np.max(a)


# In[94]:


np.argmin(a)


# In[95]:


np.argmax(a)


# In[96]:


np.sqrt(a)


# In[97]:


np.sin(a)


# ## Linspace() ==> This function returns value between a given range and with a same gap between consecutive elements.
# 

# In[108]:


import numpy as np
a= np.linspace(1,20,5)
a


# ## np.unique(arr,return_index = True, return_count = True)
# 
# returns 3 array ==> 1.the array contain unique values. 2. The array with respective index values. 3. The array with counting of frequency of each element.

# In[112]:


a = np.array([10,20,30,90,68,90,40,50])
np.unique(a,True,True)


# ## Horizontal and verticle stacking

# In[113]:


a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
a


# In[114]:


b


# In[115]:


np.hstack((a,b))


# In[116]:


np.vstack((a,b))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




