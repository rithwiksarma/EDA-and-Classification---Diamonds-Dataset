#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[5]:


dataset = pd.read_csv("C:/Users/rithw/Desktop/Sem IX/Python (PP)/Python Project/diamonds.csv")


# In[11]:


print(dataset.info())
print(dataset.describe())
dataset.isnull().sum()
dataset = dataset.fillna(dataset['carat'].value_counts().index[0])
print((dataset==0).sum())
dataset.iloc[:,6:]=dataset.iloc[:,6:].replace(0, np.NaN)
dataset = dataset.fillna(dataset['price'].value_counts().index[0])
sns.boxplot(x=dataset['carat'])
sns.boxplot(x=dataset['x'])
sns.boxplot(x=dataset['y'])
sns.boxplot(x=dataset['z'])
sns.boxplot(x=dataset['depth'])
sns.boxplot(x=dataset['table'])
sns.boxplot(x=dataset['price'])
dataset['cut'].unique()
dataset['color'].unique()
dataset['clarity'].unique()
print(dataset['clarity'].value_counts())
print(dataset['color'].value_counts())
print(dataset['cut'].value_counts())


# In[12]:


dataset.drop(dataset.loc[dataset['clarity']=='XXX'].index, inplace=True)
dataset.drop(dataset.loc[dataset['clarity']==' '].index, inplace=True)
dataset.drop(dataset.loc[dataset['cut']==0.3].index, inplace=True)
dataset.drop(dataset.loc[dataset['cut']=='Wonderful'].index, inplace=True)
dataset.drop(dataset.loc[dataset['color']== 'AAA'].index, inplace=True)
dataset.drop(dataset.loc[dataset['color']==0.3].index, inplace=True)
dataset['cut'] = pd.Categorical(dataset['cut'], ['Fair','Good','Very Good','Ideal','Premium'], ordered = True)
dataset['cut'] = dataset['cut'].cat.codes
dataset['color'] = pd.Categorical(dataset['color'], ['D','E','F','G','H','I','J'], ordered = True)
dataset['color'] = dataset['color'].cat.codes
dataset['clarity'] = pd.Categorical(dataset['clarity'], ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], ordered = True)
dataset['clarity'] = dataset['clarity'].cat.codes


# In[13]:


# regression

allCols = dataset.columns.tolist()
allCols.remove('depth')
allCols.remove('x')
allCols.remove('y')
allCols.remove('z')
allCols.remove('table')
allCols.remove('price')
print(allCols)


# In[14]:


x=dataset[allCols]
y=dataset.price
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
#predict
y_pred = model.predict(x_test)
print(y_pred)


# In[18]:


dataset.head(5)


# In[23]:


pd.crosstab(dataset.carat,dataset.cut, values=dataset.price,  aggfunc='mean').fillna('-')


# In[25]:


pd.crosstab(dataset.carat,dataset.clarity, values=dataset.price,  aggfunc='mean').fillna('-')


# In[26]:


pd.crosstab(dataset.carat,dataset.color, values=dataset.price,  aggfunc='mean').fillna('-')


# In[ ]:




