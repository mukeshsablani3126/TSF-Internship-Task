#!/usr/bin/env python
# coding: utf-8

# # Task #4: Decision Tree Algorithm
# Decision tree algorithm falls under the category of supervised learning. They can be used to solve both regression and classification problems.
# 
# Decision tree uses the tree representation to solve the problem in which each leaf node corresponds to a class label and attributes are represented on the internal node of the tree.
# 
# 
# Task
# For the given ‘Iris’ dataset, create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.
# 
# Dataset : https://drive.google.com/file/d/11Iq7YvbWZbt8VXjfm06brx66b10YiwK-/view?usp=sharing

# In[1]:


#Loading Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Loading the iris dataset


df=pd.read_csv("C:\python\Iris.csv")
df.head()


# In[3]:


df_new=df.rename(columns={'SepalLengthCm':'Sepal_Length','SepalWidthCm':'Sepal_Width',
                          'PetalLengthCm':'Petal_Length','PetalWidthCm':'Petal_Width'})
df_new.head()


# # Explore our data set

# In[4]:


df_new.shape


# # Data Preprocessing

# In[5]:


df_new.info()


# There are 4 - Numerical Features and one categorical column
# There are totally 150 rows or observations are in data

# In[6]:


# see descriptive analysis
df_new.describe()


# In[7]:


sns.jointplot(x='Sepal_Length',y= 'Sepal_Width', data=df_new)


# In[8]:


sns.countplot(x='Petal_Width' , data=df_new)


# In[9]:


# Checking null Vaules
df_new.isnull().sum()


# In[10]:


df_new=df_new.drop(['Species','Id'],axis=1)
df_new.head()


# # Outlier Analysis

# In[11]:


#select only numeric
cnames = df_new.select_dtypes(include=np.number)


# In[12]:



#plot boxplot
f , ax = plt.subplots(figsize =(15,15))
fig = sns.boxplot(data =cnames)


# In[22]:


# In[13]:



# #Detect and delete outliers from data


for i in cnames:
    print(i)
    q75, q25 = np.percentile(df_new.loc[:,i], [75,25])
    iqr = q75 - q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print(min)
    print(max)


# In[ ]:





# # Model Development

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn import tree


# In[15]:


X=df_new
y=df["Species"]


# In[16]:


# Split Train and Test Data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[17]:


X_train


# In[18]:


y_train


# In[19]:


# fit train and test data
tree1=DecisionTreeClassifier(criterion='entropy')
tree1.fit(X_train,y_train)
y_pred=tree1.predict(X_test)


# In[20]:


print(classification_report(y_test,y_pred))


# In[21]:


print(confusion_matrix(y_test,y_pred))


# In[22]:


# calculating accuracy of model
print(accuracy_score(y_pred,y_test))


# In[23]:


# Heat map analysis
sns.heatmap(confusion_matrix(y_pred,y_test),annot=True)
plt.show()


# In[24]:



cols=list(df_new.columns.values)
cols


# In[25]:


plt.figure(figsize=(14,9))
tree.plot_tree(tree1.fit(X,y),feature_names=cols,filled=True,precision=3,
              proportion=True,rounded=True)
plt.show()


# In[ ]:




