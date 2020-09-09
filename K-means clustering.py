#!/usr/bin/env python
# coding: utf-8

# # Task # 3 - To Explore Unsupervised Machine Learning
# 
# Problem Defination :
# From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.

# # Import library

# In[6]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import preprocessing
from sklearn.cluster import KMeans


# # load Iris data set

# In[13]:


df = pd.read_csv("C:\sparks foundations\Iris.csv")


# In[14]:


# getting Data
df.drop('Id',axis=1,inplace=True)
df.head()# See the first 5 rows


# In[15]:


df['Species'].value_counts()


# # Data Preprocessing

# In[16]:


# Descriptive analysis of data
df.describe()


# In[17]:


# Data type of variables
df.info()


# In[18]:


# Divide the data into inputs and labels

X= df.drop('Species',axis=1)
y = df['Species']


# In[19]:


X.head()


# # Visualization

# In[23]:


#Scatter plot

sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.scatter(data=X,x='SepalLengthCm',y='SepalWidthCm')
plt.xlabel('Length of sepal')
plt.ylabel('Width of sepal');


# # Feature scaling

# In[25]:


#Standardization of variables
sns.distplot(df['SepalLengthCm']);


# # here we can see above figure our data is normally distributed then we use the standardization method so we  used scaling the variables.

# In[26]:


# Scale the variables

X_scaled = preprocessing.scale(X)
X_scaled[:10]


# In[38]:


# Finding the optimum number of clusters for k-means classification

x = df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# 
# You can clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.
# Based on the elbow curve it seems like we can plot our graph with 2 , 3 and 5 no. of clusters .

# In[39]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[40]:


# second cluster

kmeans_2 = KMeans(2)
kmeans_2.fit(X_scaled)

cl_2 = X.copy()

cl_2['pred'] = kmeans_2.fit_predict(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(cl_2['SepalLengthCm'], cl_2['SepalWidthCm'], c= cl_2['pred'], cmap = 'rainbow');


# In[41]:



# 3 cluster

kmeans_3 = KMeans(3)
kmeans_3.fit(X_scaled)

cl_3 = X.copy()

cl_3['pred'] = kmeans_3.fit_predict(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(cl_3['SepalLengthCm'], cl_3['SepalWidthCm'], c= cl_3['pred'], cmap = 'rainbow');


# In[45]:


# 5 cluster

kmeans_5 = KMeans(5)
kmeans_5.fit(X_scaled)

cl_5 = X.copy()

cl_5['pred'] = kmeans_5.fit_predict(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(cl_5['SepalLengthCm'], cl_5['SepalWidthCm'], c= cl_5['pred'], cmap = 'rainbow');


# In[46]:


# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# This is K-means workshop

# In[ ]:




