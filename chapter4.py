#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


n_rows = 300000
df = pd.read_csv("train.csv", nrows=n_rows)


# In[6]:


print(df.head(5))


# In[7]:


Y = df['click'].values


# In[9]:


X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'],axis=1).values


# In[14]:


print(X.shape)


# In[39]:


n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]


# In[40]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')


# In[41]:


X_train_enc = enc.fit_transform(X_train)
X_train_enc[0]
print(X_train_enc[0])


# In[42]:


X_test_enc = enc.transform(X_test)


# In[43]:


from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': [3, 10, None]}


# In[44]:


decision_tree = DecisionTreeClassifier(criterion='gini',min_samples_split=30)
from sklearn.model_selection import GridSearchCV


# In[45]:


grid_search = GridSearchCV(decision_tree, parameters,n_jobs=-1, cv=3, scoring='roc_auc')


# In[50]:


grid_search.fit(X_train_enc, Y_train)


# In[51]:


print(grid_search.best_params_)


# In[57]:


import numpy as np
decision_tree_best = grid_search.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]
from sklearn.metrics import roc_auc_score
print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test,pos_prob):.3f}')


# In[59]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100,criterion='gini', min_samples_split=30,n_jobs=-1)


# In[63]:


grid_search = GridSearchCV(random_forest, parameters,n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)


# In[65]:


print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test,pos_prob):.3f}')


# In[66]:


pos_prob = np.zeros(len(Y_test))
click_index = np.random.choice(len(Y_test),int(len(Y_test) * 51211.0/300000),replace=False)
pos_prob[click_index] = 1
print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test,pos_prob):.3f}')


# In[67]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100,criterion='gini', min_samples_split=30,n_jobs=-1)


# In[70]:


grid_search = GridSearchCV(random_forest, parameters,n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)


# In[82]:


decision_tree_best = grid_search.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]
print('The ROC AUC on testing set is:{0:.3f}'.format(roc_auc_score(Y_test, pos_prob)))


# In[ ]:




