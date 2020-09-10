#!/usr/bin/env python
# coding: utf-8

# # Task #5: To explore Business Analytics
# Task
# Perform ‘Exploratory Data Analysis’ on the provided dataset ‘SampleSuperstore’ You are the business owner of the retail firm and want to see how your company is performing. You are interested in finding out the weak areas where you can work to make more profit. What all business problems you can derive by looking into the data? You can choose any of the tool of your choice (Python/R/Tableau/PowerBI/Excel)
# 
# Dataset: https://drive.google.com/file/d/1lV7is1B566UQPYzzY8R2ZmOritTW299S/view
# 
# Import all libraries

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


# Load Data Set
business=pd.read_csv('C:/sparks foundations/SampleSuperstore.csv')
business.head()


# # EDA Analysis

# In[17]:


business.shape


# In[18]:


business.info()


# In[20]:


# descriptive analysis
df.describe()


# In[23]:


sns.jointplot(x='Sales',y= 'Quantity', data=df)


# In[24]:


sns.countplot(x='Profit' , data=df)


# In[25]:


# Checking Null Values
df.isnull().any()


# In[21]:


df.duplicated().sum()


# In[22]:


df.drop_duplicates(keep='first',inplace=True)
df


# In[26]:


# Correlation analysis
corr=df.corr()
corr


# In[27]:


sns.heatmap(corr,annot=True)


# In[28]:


sns.countplot(df['Ship Mode'])
plt.show()


# In[29]:


sns.countplot(df['Segment'])
plt.show()


# In[30]:


# plot histogramme
df.hist(figsize=(10,10),bins=50)
plt.show()


# In[31]:


plt.figure(figsize=(12,7))
plt.scatter(x=df['Sales'],y=df['Profit'],c='m')
plt.title("Sale VS Profit")
plt.xlabel('Total Sale')
plt.ylabel("Profit")
plt.show()


# In[32]:


plt.figure(figsize=(12,8))
sns.countplot(df['State'],order=(df['State'].value_counts().head(30)).index)
plt.xticks(rotation=90)
plt.show()


# In[33]:


plt.figure(figsize=(10,8))
df['Sub-Category'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()


# In[34]:


# total profit and sales
plt.figure(figsize=(11,7))
df.groupby('Sub-Category')['Profit','Sales'].agg(['sum']).plot.bar()
plt.show()


# In[35]:


# Counts of Sub-Category Region-Wise
plt.figure(figsize=(12,7))
sns.countplot(x='Sub-Category',hue='Region',data=df)
plt.xticks(rotation=90)
plt.show()


# Here we can easily conclude how much effect our data set variable in a business with various graphs and we can compare overall statistics our data in terms of retailer business analytics.
