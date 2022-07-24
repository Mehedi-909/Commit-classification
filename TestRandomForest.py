#!/usr/bin/env python
# coding: utf-8

# In[162]:


# All purpose 
import pandas as pd
import numpy as np
import re
from numpy import mean
from numpy import std
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn ML
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import tree
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
import sys
sys._enablelegacywindowsfsencoding()
pd.set_option('display.max_rows', None)


# In[98]:


#fields = ['comment', 'label']
data = pd.read_csv("E:\Semester 8\Software Maintenance\Assignment\dataset.csv", skipinitialspace=True,delimiter="#")
#commit_text= df['comment'].dropna()

print("Data shape:", data.shape)


# In[106]:


#Keywords
X = data[["add","allow","bug","chang","error","fail","fix","implement","improv","issu","method","new","npe","refactor","remov","report","set","support","test","use"]]
#X = data["add","allow"]
y = data["label"]


# In[107]:


#Changes
X = data[["ADDING_ATTRIBUTE_MODIFIABILITY","ADDING_CLASS_DERIVABILITY","ADDING_METHOD_OVERRIDABILITY","ADDITIONAL_CLASS","ADDITIONAL_FUNCTIONALITY","ADDITIONAL_OBJECT_STATE","ALTERNATIVE_PART_DELETE","ALTERNATIVE_PART_INSERT","ATTRIBUTE_RENAMING","ATTRIBUTE_TYPE_CHANGE","CLASS_RENAMING","COMMENT_DELETE","COMMENT_INSERT","COMMENT_MOVE","COMMENT_UPDATE","CONDITION_EXPRESSION_CHANGE","DECREASING_ACCESSIBILITY_CHANGE","DOC_DELETE","DOC_INSERT","DOC_UPDATE","INCREASING_ACCESSIBILITY_CHANGE","METHOD_RENAMING","PARAMETER_DELETE","PARAMETER_INSERT","PARAMETER_ORDERING_CHANGE","PARAMETER_RENAMING","PARAMETER_TYPE_CHANGE","PARENT_CLASS_CHANGE","PARENT_CLASS_DELETE","PARENT_CLASS_INSERT","PARENT_INTERFACE_CHANGE","PARENT_INTERFACE_DELETE","PARENT_INTERFACE_INSERT","REMOVED_CLASS","REMOVED_FUNCTIONALITY","REMOVED_OBJECT_STATE","REMOVING_ATTRIBUTE_MODIFIABILITY","REMOVING_CLASS_DERIVABILITY","REMOVING_METHOD_OVERRIDABILITY","RETURN_TYPE_CHANGE","RETURN_TYPE_DELETE","RETURN_TYPE_INSERT","STATEMENT_DELETE","STATEMENT_INSERT","STATEMENT_ORDERING_CHANGE","STATEMENT_PARENT_CHANGE","STATEMENT_UPDATE","UNCLASSIFIED_CHANGE"]]
#X = data["add","allow"]
y = data["label"]


# In[121]:


#Combined
X = data[["ADDING_ATTRIBUTE_MODIFIABILITY","ADDING_CLASS_DERIVABILITY","ADDING_METHOD_OVERRIDABILITY","ADDITIONAL_CLASS","ADDITIONAL_FUNCTIONALITY","ADDITIONAL_OBJECT_STATE","ALTERNATIVE_PART_DELETE","ALTERNATIVE_PART_INSERT","ATTRIBUTE_RENAMING","ATTRIBUTE_TYPE_CHANGE","CLASS_RENAMING","COMMENT_DELETE","COMMENT_INSERT","COMMENT_MOVE","COMMENT_UPDATE","CONDITION_EXPRESSION_CHANGE","DECREASING_ACCESSIBILITY_CHANGE","DOC_DELETE","DOC_INSERT","DOC_UPDATE","INCREASING_ACCESSIBILITY_CHANGE","METHOD_RENAMING","PARAMETER_DELETE","PARAMETER_INSERT","PARAMETER_ORDERING_CHANGE","PARAMETER_RENAMING","PARAMETER_TYPE_CHANGE","PARENT_CLASS_CHANGE","PARENT_CLASS_DELETE","PARENT_CLASS_INSERT","PARENT_INTERFACE_CHANGE","PARENT_INTERFACE_DELETE","PARENT_INTERFACE_INSERT","REMOVED_CLASS","REMOVED_FUNCTIONALITY","REMOVED_OBJECT_STATE","REMOVING_ATTRIBUTE_MODIFIABILITY","REMOVING_CLASS_DERIVABILITY","REMOVING_METHOD_OVERRIDABILITY","RETURN_TYPE_CHANGE","RETURN_TYPE_DELETE","RETURN_TYPE_INSERT","STATEMENT_DELETE","STATEMENT_INSERT","STATEMENT_ORDERING_CHANGE","STATEMENT_PARENT_CHANGE","STATEMENT_UPDATE","UNCLASSIFIED_CHANGE","add","allow","bug","chang","error","fail","fix","implement","improv","issu","method","new","npe","refactor","remov","report","set","support","test","use"]]
#X = data["add","allow"]
y = data["label"]


# In[122]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[123]:


#Random Forest
rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
# predictions

rf_pred_train = rfc.predict(X_train)
rf_pred_test = rfc.predict(X_test)
accuracy = metrics.accuracy_score(y_test, rf_pred_test)
accuracy
#print("F1 score: {}%".format(round(metrics.f1_score(y_test, rf_pred_test) * 100,3)))


# In[114]:


#cross-validation
print (np.mean(cross_val_score(rfc, X_train, y_train, cv=10)))


# In[173]:


#kappa
cohen_score = cohen_kappa_score(y_test, rf_pred_test)
print('Kappa: %.3f' % (cohen_score*100))


# In[127]:


matrix = confusion_matrix(y_test, rf_pred_test)
matrix


# In[143]:


report = classification_report(y_test, rf_pred_test)
print(report)


# In[149]:


precision,recall,fscore,support=score(y_test, rf_pred_test)
print ('Precision : {}'.format(precision*100))
print ('Recall    : {}'.format(recall*100))


# In[184]:


#GBM
model = GradientBoostingClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f' % (mean(n_scores)*100))


# In[183]:


#J48
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf_pred_train = clf.predict(X_train)
clf_pred_test = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, clf_pred_test)
print('Accuracy: %.3f' % (accuracy*100))
#cohen_score = cohen_kappa_score(y_test, clf_pred_test)
#print('Kappa: %.3f' % (cohen_score*100))


# In[ ]:





# In[ ]:




