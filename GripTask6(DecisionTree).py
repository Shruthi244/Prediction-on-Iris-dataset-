#!/usr/bin/env python
# coding: utf-8

# In[28]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics


# In[2]:


data=pd.read_csv("C:\\Users\\JAYALAKSHMI\\Downloads\\Iris.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.tail()


# In[6]:


data.columns


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


data.isnull().sum()


# In[12]:


features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

# Create Features matrix
x=data.loc[:,features].values
y=data.Species


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# # Decision tree classifier

# In[14]:


clf=DecisionTreeClassifier(max_depth=2,random_state=0)
clf.fit(x_train,y_train)


# In[15]:


clf.predict(x_test[0:1])


# # Measuring the performance of the model

# In[16]:


score=clf.score(x_test,y_test)
print(score)


# In[17]:


print(metrics.classification_report(y_test,clf.predict(x_test)))


# >Setosa flower accuracy is 1.0
# >>Versicoor precision is 0.83
# >>>Virginica precision is 0.86

# In[20]:


cm=metrics.confusion_matrix(y_test,clf.predict(x_test))
plt.figure(figsize=(7,7))
sns.heatmap(cm,annot=True,fmt=".0f",linewidth=0.5,square=True,cmap='Blues');
plt.ylabel('Actual Label',fontsize=17);
plt.xlabel('Predicted Label',fontsize=17);
plt.title('Accuracy score: {}'.format(score),size=17);
plt.tick_params(labelsize=15)


# # Finding the optimal max_depth

# In[21]:


max_depth_range=list(range(1,6))
accuracy=[]

for depth in max_depth_range:
    clf=DecisionTreeClassifier(max_depth=depth,random_state=0)
    clf.fit(x_train,y_train)
    score=clf.score(x_test,y_test)
    accuracy.append(score)


# In[22]:


# Plotting accuracy score depth wise

fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,17));
ax.plot(max_depth_range,accuracy,lw=2,color='k')

ax.set_xlim([1,5])
ax.set_ylim([.50,1.00])
ax.grid(True,axis='both',zorder=0,linestyle=':',color='k')

ax.tick_params(labelsize=18)
ax.set_xticks([1,2,3,4,5])
ax.set_xlabel('max_depth',fontsize=24)
ax.set_ylabel('Accuracy',fontsize=24)
fig.tight_layout()


# # Decision tree Visualization

# In[25]:


fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(7,4),dpi=150)
tree.plot_tree(clf);


# # Making Decision tree more interpretable

# In[26]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa','versicolor','virginica']


# In[27]:


fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(7,4),dpi=300)
tree.plot_tree(clf,feature_names=fn,class_names=cn,filled=True);

