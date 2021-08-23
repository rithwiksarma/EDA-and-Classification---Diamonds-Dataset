
# coding: utf-8

# In[142]:



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd
from sklearn.preprocessing import Imputer


# In[143]:



dataset = pd.read_csv('C:/Users/rithw/Desktop/Sem IX/Python (PP)/Python Project/diamonds.csv')


# In[144]:


get_ipython().magic('matplotlib inline')


# In[145]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)]
    return df_out[col_name]

m=dataset.drop('cut', axis=1)
n=m.drop('color', axis=1)
o=n.drop('clarity', axis=1)
print(o.info())

colNames=o.columns
print("columns with outliers")
for colName in colNames:
    colvalues = o[colName].values
    #print('column12:', colvalues)
    print('column:', colName)
    outValues=remove_outlier(o,colName)
    print(outValues)
    G=outValues.count()
    print(G)
#print(o.carat.describe())


# In[146]:


print(dataset.info())
print(dataset.describe())

print(dataset.isnull().sum())
#dataset = dataset.fillna(dataset['carat'].value_counts().index[0])
print((dataset==0).sum())

#dataset.iloc[:,6:]=dataset.iloc[:,6:].replace(0, np.NaN)
#dataset = dataset.fillna(dataset['price'].value_counts().index[0])

dataset['cut'].unique()
dataset['color'].unique()
dataset['clarity'].unique()
print(dataset['clarity'].value_counts())
print(dataset['color'].value_counts())
print(dataset['cut'].value_counts())


# In[147]:


sns.boxplot(x=dataset['carat'])


# In[102]:




# X = iqrOutCount(dataset['carat'])
# X


# In[148]:


sns.boxplot(x=dataset['x'])


# In[149]:


sns.boxplot(x=dataset['y'])


# In[ ]:


sns.boxplot(x=dataset['z'])


# In[ ]:


sns.boxplot(x=dataset['depth'])


# In[ ]:


sns.boxplot(x=dataset['table'])


# In[ ]:


sns.boxplot(x=dataset['price'])


# In[150]:


dataset.drop(dataset.loc[dataset['clarity']=='XXX'].index, inplace=True)
dataset.drop(dataset.loc[dataset['clarity']==' '].index, inplace=True)
dataset.drop(dataset.loc[dataset['cut']==0.3].index, inplace=True)
dataset.drop(dataset.loc[dataset['cut']=='Wonderful'].index, inplace=True)
dataset.drop(dataset.loc[dataset['color']== 'AAA'].index, inplace=True)
dataset.drop(dataset.loc[dataset['color']==0.3].index, inplace=True)
dataset['cut'].fillna('null entry', inplace = True)
dataset['cut'] = pd.Categorical(dataset['cut'], ['null entry','Fair','Good','Very Good','Ideal','Premium'], ordered = True)

dataset['cut'] = dataset['cut'].cat.codes
dataset['color'] = pd.Categorical(dataset['color'], ['D','E','F','G','H','I','J'], ordered = True)
dataset['color'] = dataset['color'].cat.codes
dataset['clarity'] = pd.Categorical(dataset['clarity'], ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], ordered = True)
dataset['clarity'] = dataset['clarity'].cat.codes
#dataset = dataset.fillna(0)
 
dataset['clarity'].fillna(0, inplace = True)
dataset['carat'].fillna(0, inplace = True) 
dataset['price'].fillna(0, inplace = True)


# In[151]:


#dataset.loc[dataset.price.isnull()==True]
df = dataset.loc[dataset.price==0]
df

# dataset.loc[dataset.price==np.NaN]


# In[153]:


x = dataset.drop(dataset.loc[dataset.price==0].index,axis=0)


# In[154]:




# dataset.iloc[:,6:]=dataset.iloc[:,6:].replace(0, np.NaN)
#dataset = dataset.fillna(dataset.price=0)


# In[155]:


# regression

allCols = dataset.columns.tolist()
allCols.remove('depth')
allCols.remove('x')
allCols.remove('y')
allCols.remove('z')
allCols.remove('table')
allCols.remove('price')
allCols
print(allCols)


# In[156]:


x1=x[allCols]
y1=x.price
#y_pred=df.price
ynew=df[allCols]

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x1,y1)

#predict
y_pred = model.predict(ynew)
print(y_pred)






# In[157]:


# classification
allCols2 = dataset.columns.tolist()
allCols2.remove('depth')
allCols2.remove('x')
allCols2.remove('y')
allCols2.remove('z')
allCols2.remove('table')
allCols2.remove('cut')
print(allCols2)


# In[158]:


df1 = dataset.loc[dataset['cut']==0]
df1


# In[159]:


p = dataset.drop(dataset.loc[dataset.cut==0].index,axis=0)


# In[160]:


xp=p[allCols2]
yp=p.cut
y_pred2=df1.cut
ynew2=df1[allCols2]
#print(y_pred2)
#print(ynew2)
#print(xp)
#print(yp)
#x2=dataset[allCols2]
#y2=dataset.cut
#from sklearn.model_selection import train_test_split
#x_train2, x_test2, y_train2, y_test2 = train_test_split(x2,y2, test_size=0.25)
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier()
model2.fit(xp,yp)
#predict
y_pred2 = model2.predict(ynew2)
print(y_pred2)


# In[161]:


pd.crosstab(dataset.carat,dataset.cut, values=dataset.price,  aggfunc='mean').fillna('-')


# In[25]:




# In[162]:


pd.crosstab(dataset.carat,dataset.clarity, values=dataset.price,  aggfunc='mean').fillna('-')


# In[26]:




# In[163]:


pd.crosstab(dataset.carat,dataset.color, values=dataset.price,  aggfunc='mean').fillna('-')


# In[164]:



#print(dataset.isnull().sum())

print((dataset==0).x.sum())
Mean1 = int(dataset['x'].mean())
print("Mean of column X :",Mean1)
print((dataset==0).y.sum())
Mean2 = int(dataset['y'].mean())
print("Mean of column Y :",Mean2)
print((dataset==0).z.sum())
Mean3=(dataset['z'].mean())
print("Mean of column Z :",Mean3)


# In[165]:


# dataset['x'] = np.where((dataset==0).x.sum(), Mean1, dataset['x'])

dataset[dataset.x==0] = dataset[dataset.x==0].replace(0,Mean1)

print((dataset==0).x.sum())

dataset[dataset.y==0] = dataset[dataset.y==0].replace(0,Mean2)

print((dataset==0).y.sum())

dataset[dataset.z==0] = dataset[dataset.z==0].replace(0,Mean3)

print((dataset==0).z.sum())


dataset['Depth_Percent']=dataset['z']/((dataset['x']+dataset['y'])/2)


print(dataset[dataset['Depth_Percent'] > 5])


# In[58]:





# In[166]:


plt.hist(dataset['price'], bins = 1000,  color = 'red', alpha=0.5)
plt.xlim(xmin=0, xmax = 20000)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distribution of Price")
plt.show()


# In[ ]:


sns.distplot(dataset['cut'], kde=False, rug=True);
plt.ylabel("Count")
plt.xlabel("Cut")
plt.title("Distribution of Cut")
plt.show()

# In[ ]:


plt.figure()
sns.lmplot(x='price', y='carat', data=dataset,
           fit_reg=True,line_kws={'color':'red'})
# tweak using matplotlib
plt.ylim(0, 6)
plt.xlim(0,20000 )
plt.title('Relationship between Price and Carat')
#plt.ylabel('Y axis')
#plt.xlabel('X axis')
# good practice
plt.show()


# In[ ]:


plt.figure()
sns.lmplot(x='price', y='x', data=dataset,
           fit_reg=True,line_kws={'color':'red'})
# tweak using matplotlib
plt.ylim(0, 12)
plt.xlim(0,20000)
plt.title('Relationship between Price and X ')
#plt.ylabel('Y axis')
#plt.xlabel('X axis')
# good practice
plt.show()


# In[ ]:


sns.boxplot('color', 'price', data=dataset)
plt.ylim(0,20000)

