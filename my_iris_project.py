#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# ### Importing data set

# In[2]:


iris = pd.read_csv('IRIS.csv')
iris.head()


# In[3]:


iris.shape


# In[4]:


iris.isnull().sum()


# In[5]:


## In this data set don't have any missing values


# In[6]:


iris.corr()


# In[7]:


sns.heatmap(iris.corr())


# In[8]:


iris.describe()


# In[9]:


iris.species.unique()


# In[10]:


### plot the data to understand the relation between species and sepal length.


# In[11]:


sns.barplot(x='species',y='sepal_length',data=iris)


# In[12]:


sns.barplot(x='species',y='sepal_width',data=iris)


# In[13]:


sns.barplot(x='species',y='petal_length',data=iris)


# In[14]:


sns.barplot(x='species',y='petal_width',data=iris)


# In[15]:


### Iris virginica has the highest sepal length and iris setosa has highest sepal width.
### Iris virginiva has highest petal_length & petal width as well.


# ### Deal with categorical values

# In[16]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(iris.species)
iris['species'] = le.fit_transform(iris.species)


# In[17]:


iris.head()


# ### Declare dependent & Independent variable

# In[18]:


x = iris.drop('species',axis=1)
y = iris['species']


# In[19]:


x.shape


# In[20]:


y.shape


# In[21]:


from sklearn.preprocessing import Normalizer


# In[22]:


scaler = Normalizer()
X = scaler.fit_transform(x)
X= pd.DataFrame(X, columns = x.columns)
X.head()


# ### Split the Data

# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)


# In[24]:


X_train.shape


# In[25]:


X_test.shape


# In[26]:


y_train.shape


# In[27]:


y_test.shape


# ### Train & predict the model

# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
logmodel = LogisticRegression(random_state=0)
logmodel.fit(X_train,y_train)


# In[29]:


logpred = logmodel.predict(X_test)


# In[30]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[31]:


accuracy = confusion_matrix(y_test,logpred)
accuracy


# In[32]:


accuracy = accuracy_score(y_test,logpred)
accuracy


# ### Try with KNeighbors Classifier

# In[33]:


from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=13)
knc.fit(X_train,y_train)


# In[34]:


kpred = knc.predict(X_test)


# In[35]:


accuracy = confusion_matrix(y_test,kpred)
accuracy


# In[36]:


accuracy = accuracy_score(y_test,kpred)
accuracy


# In[37]:


import pickle
pickle.dump(knc, open('model.pkl','wb'))


# In[ ]:




