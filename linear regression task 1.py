#!/usr/bin/env python
# coding: utf-8

# # SIMPLE LINEAR REGRESSION PYTHON WITH SCIKIT LEARN
# In given task we have to predict the percentage of marks expected by the student based upon the number of hours they studied.In this task only two variables are involved.

# In[29]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[30]:


#Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
#print("Data imported successfully")

s_data.head(10)


# # #DATA VISUALIZATION
# Now let's plot a graph of our data so that it will give us clear idea about data.

# In[31]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # LINEAR REGRESSION MODEL
# Preparing the data
# Divided the datast into attributes (inputs) and labels (outputs).

# In[32]:


#Splitting training and testing data
x=s_data.iloc[:,:-1].values
y=s_data.iloc[:,1].values
x_train, x_test, y_train, y_test= train_test_split(x, y,train_size=0.80,test_size=0.20,random_state=0)


# # Training the model

# In[33]:


from sklearn.linear_model import LinearRegression
linearRegressor= LinearRegression()
linearRegressor.fit(x_train, y_train)
y_predict= linearRegressor.predict(x_train)


# # Training the Algorithm
# Now the spliting of our data into training and testing sets is done, now it's time to train our algorithm.

# In[34]:



regressor = LinearRegression()  
regressor.fit(x_train, y_train) 

print("Training complete.")


# In[35]:


# Plotting the regression line
line = regressor.coef_*x+regressor.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# # Predicting the results

# In[36]:


print(x_test) # Testing data - In Hours
y_pred = regressor.predict(x_test) # Predicting the scores


# In[37]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[38]:


#Let's predict the score for 9.25 hours
print('Score of student who studied for 9.25 hours a dat', regressor.predict([[9.25]]))


# # Model evaluation matrics
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[40]:


#Checking the efficiency of model
from sklearn import metrics
print("Mean Squared Error:",metrics.mean_squared_error(y_test,y_pred))
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




