#!/usr/bin/env python
# coding: utf-8

# In[265]:


# All purpose 
import pandas as pd
import numpy as np
import re
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn ML
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn import preprocessing
import sys
sys._enablelegacywindowsfsencoding()


# In[270]:


fields = ['comment', 'label']
data2 = pd.read_csv("E:\Semester 8\Software Maintenance\Assignment\d2.csv", skipinitialspace=True, usecols=fields,delimiter="#")
commit_text= df['comment'].dropna()

# print basic information
#print("Data shape:", data2.shape)
data = data2.head(n=15)
print("Data shape:", data.shape)


# In[271]:


print(*data["label"].unique(), sep='\n')


# In[272]:


X = data["comment"].values
y = data['label'].values


y


# In[282]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[283]:


rfc = RandomForestClassifier()
le = preprocessing.LabelEncoder()
le.fit(X_train.reshape(-1, 1))
le.fit(y_train.reshape(-1, 1))
le.transform(X_train.reshape(-1, 1))
rfc.fit(X_train.reshape(-1, 1),y_train.reshape(-1, 1))
# predictions

y_pred = rfc.predict(X_train)
rfc_pred = rfc.predict(X_test.reshape)
print("F1 score: {}%".format(round(metrics.f1_score(y_test, rfc_pred) * 100,3)))


# In[ ]:




