#!/usr/bin/env python
# coding: utf-8

# # Task #5: To explore Business Analytics
# Task
# Perform ‘Exploratory Data Analysis’ on the provided dataset ‘SampleSuperstore’ You are the business owner of the retail firm and want to see how your company is performing. You are interested in finding out the weak areas where you can work to make more profit. What all business problems you can derive by looking into the data? You can choose any of the tool of your choice (Python/R/Tableau/PowerBI/Excel)
# 
# Dataset: https://drive.google.com/file/d/1lV7is1B566UQPYzzY8R2ZmOritTW299S/view
# 
# Import all libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load Data Set
business=pd.read_csv('C:/sparks foundations/SampleSuperstore.csv')
business.head()


# # EDA Analysis

# In[3]:


business.shape


# In[4]:


business.info()


# In[5]:


# descriptive analysis
business.describe()


# In[6]:


sns.jointplot(x='Sales',y= 'Quantity', data=business)


# In[7]:


sns.countplot(x='Profit' , data=business)


# In[8]:


# Checking Null Values
business.isnull().any()


# In[9]:


business.duplicated().sum()


# In[10]:


business.drop_duplicates(keep='first',inplace=True)
business


# In[11]:


# Correlation analysis
corr=business.corr()
corr


# In[12]:


sns.heatmap(corr,annot=True)


# In[13]:


sns.countplot(business['Ship Mode'])
plt.show()


# In[14]:


sns.countplot(business['Segment'])
plt.show()


# In[16]:


# plot histogramme
business.hist(figsize=(10,10),bins=50)
plt.show()


# In[17]:


plt.figure(figsize=(12,7))
plt.scatter(x=business['Sales'],y=business['Profit'],c='m')
plt.title("Sale VS Profit")
plt.xlabel('Total Sale')
plt.ylabel("Profit")
plt.show()


# In[19]:


plt.figure(figsize=(12,8))
sns.countplot(business['State'],order=(business['State'].value_counts().head(30)).index)
plt.xticks(rotation=90)
plt.show()


# In[20]:


plt.figure(figsize=(10,8))
business['Sub-Category'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()


# In[21]:


# total profit and sales
plt.figure(figsize=(11,7))
business.groupby('Sub-Category')['Profit','Sales'].agg(['sum']).plot.bar()
plt.show()


# In[22]:


# Counts of Sub-Category Region-Wise
plt.figure(figsize=(12,7))
sns.countplot(x='Sub-Category',hue='Region',data=business)
plt.xticks(rotation=90)
plt.show()


# Here we can easily conclude how much effect our data set variable in a business with various graphs and we can compare overall statistics our data in terms of retailer business analytics.
