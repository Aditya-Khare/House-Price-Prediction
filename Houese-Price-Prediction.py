#!/usr/bin/env python
# coding: utf-8

# In[1]:


# House Price Prediction 
# using Linear Regression Suprevised Machine Learning Algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# In[2]:


# Check out the Data
USAhousing = pd.read_csv('USA_Housing.csv')
USAhousing.head()


# In[3]:


USAhousing.info()


# In[4]:


USAhousing.describe()


# In[5]:


USAhousing.columns


# In[6]:


# Exploratory Data Analysis (EDA)


# In[7]:


sns.pairplot(USAhousing)


# In[8]:


sns.distplot(USAhousing['Price'])


# In[9]:


sns.heatmap(USAhousing.corr(), annot=True)


# In[10]:


# Training a Linear Regression Model


# In[11]:


# X and y arrays
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# In[12]:


# Train Test Split


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[15]:


from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('Mean Aabsolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)
    print('R2 Square', r2_square)
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# In[16]:


# Preparing Data For Linear Regression


# In[17]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)


# In[18]:


# Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)


# In[19]:


# Model Evaluation


# In[20]:


# print the intercept
print(lin_reg.intercept_)


# In[21]:


coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[22]:


# Predictions from our Model
pred = lin_reg.predict(X_test)
plt.scatter(y_test, pred)


# In[23]:


# Residual Histogram
sns.distplot((y_test - pred), bins=50);


# In[24]:


# Regression Evaluation Metrics
test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)


# In[25]:


results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df


# In[ ]:




