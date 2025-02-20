#!/usr/bin/env python
# coding: utf-8

# # Salary prediction using simple linear regression and polynomial regression

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


ds=pd.read_csv('Salary.csv')


# In[3]:


ds


# Extracting both the columns into different variables x and y
The iloc() function in python is defined in the Pandas module that helps us to select a specific row or column from the data set
# In[4]:


x=ds.iloc[:,:1].values
#x (to display independent vaiable row(experience))


# In[5]:


y=ds.iloc[:,1:].values
#y(dependent varaible i.e salary)

 Now we can visualize our data using matplot (plot the values for x and y)
 why should we visualize our data??
 To select suitable model for our project
# In[6]:


fig=plt.figure()              #creates a new figure
ax=fig.add_axes([0,0,1,1])    #adds axes to the figure
ax.scatter(x,y,color='b')    #scatter y vs x with given colour

Here we observe the pattern is somewhat linear or may be similar to polynomial
# # Train and Test

# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

.train_test_split=Split arrays or matrices into random train and test subsets.
.text_size=0.2 here 0.2 defines that 20% of dataset is for testing and remaining for training
  we have to test our dataset to find accuracy of our model
  
.random size= can be any number(how random the random can be splitted==random selection)
# In[8]:


from sklearn.linear_model import LinearRegression 


# In[9]:


linear_regressor=LinearRegression()


# # linear regression training

# In[10]:


linear_regressor.fit(x_train,y_train)


# # Prediction and testing

# In[11]:


y_pred=linear_regressor.predict(x_test)
y_pred


# In[12]:


y_test

Drawing the graph to see best fitting line 
# In[13]:


plt.scatter(x,y,color='b')
plt.plot(x,linear_regressor.predict(x),color='red')

We may have some dissmilarities in linear regression model for few predictions lets check with polynomial model
# In[14]:


from sklearn.preprocessing import PolynomialFeatures


# In[15]:


#coverting independent variable into polynomial data
poly=PolynomialFeatures(degree=2)#two varaibles
x_poly=poly.fit_transform(x)


# In[16]:


poly_regressor=LinearRegression()
poly_regressor.fit(x_poly,y)


# In[17]:


plt.scatter(x,y,color='b')
plt.plot(x,poly_regressor.predict(poly.fit_transform(x)),color='red')


# In[18]:


y_pred1=poly_regressor.predict(poly.fit_transform(x))
y_pred1 #predicted values


# In[19]:


y #actual values


# In[20]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Linear regression model evaluation
mae_linear = mean_absolute_error(y_test, y_pred)  # Corrected: use y_pred from linear regression
mse_linear = mean_squared_error(y_test, y_pred)
r2_linear = r2_score(y_test, y_pred)

# Print evaluation results
print("Linear Regression Evaluation:")
print(f"Mean Absolute Error: {mae_linear:.4f}")
print(f"Mean Squared Error: {mse_linear:.4f}")
print(f"R² Score: {r2_linear:.4f}\n")


# In[21]:


y_pred_poly = poly_regressor.predict(poly.transform(x_test))
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Print evaluation results for Polynomial Regression
print("Polynomial Regression Evaluation:")
print(f"Mean Absolute Error: {mae_poly:.4f}")
print(f"Mean Squared Error: {mse_poly:.4f}")
print(f"R² Score: {r2_poly:.4f}")

👉 Linear Regression is better if the data follows a straight-line trend.
👉 Polynomial Regression is better if the relationship is curved but should be used cautiously to avoid overfitting.

Plot your data (scatter plot) – If it looks linear, use Linear Regression. If it's curved, try Polynomial Regression.

Evaluate R² Score – Compare both models. If Polynomial Regression significantly improves accuracy, then it’s the better choice.

Check for Overfitting – If Polynomial Regression performs well on training data but poorly on test data, it may be overfitting.

# # --------------------------------------------THE END------------------------------------------------------

# In[ ]:




