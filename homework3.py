#!/usr/bin/env python
# coding: utf-8

# In[143]:


from sklearn.datasets import fetch_lfw_people
face_data = fetch_lfw_people(min_faces_per_person=80)


# In[144]:


X = face_data.data
Y = face_data.target
print('Input data size :', X.shape)
print('Output data size :', Y.shape)


# In[145]:


print('Label names:', face_data.target_names)


# In[140]:


for i in range(5):
    print(f'Class {i} has {(Y == i).sum()} samples.')


# In[146]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 4)
for i, axi in enumerate(ax.flat):
    axi.imshow(face_data.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],xlabel=face_data.target_names[face_data.target[i]])


# In[107]:



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    random_state=42)


# In[49]:


from sklearn.svm import SVC
clf = SVC(class_weight='balanced', random_state=42)


# In[50]:


parameters = {'C': [0.1, 1, 10],
              'gamma': [1e-07, 1e-08, 1e-06],
              'kernel' : ['rbf', 'linear'] }


# In[52]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)


# In[53]:


grid_search.fit(X_train, Y_train)


# In[55]:


print('The best model:\n', grid_search.best_params_) 


# In[58]:


print('The best averaged performance:', grid_search.best_score_)


# In[59]:


clf_best = grid_search.best_estimator_
pred = clf_best.predict(X_test)


# In[63]:


from sklearn.metrics import classification_report
print(f'The accuracy is: {clf_best.score(X_test, Y_test)*100:.1f}%')


# In[64]:


print(classification_report(Y_test, pred,
                            target_names=face_data.target_names))


# In[65]:


from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=42)
svc = SVC(class_weight='balanced', kernel='rbf',random_state=42)
from sklearn.pipeline import Pipeline
model = Pipeline([('pca', pca),
                  ('svc', svc)])


# In[66]:


parameters_pipeline = {'svc__C': [1, 3, 10],
                       'svc__gamma': [0.001, 0.005]}
grid_search = GridSearchCV(model, parameters_pipeline)
grid_search.fit(X_train, Y_train)


# In[67]:


print('The best model:\n', grid_search.best_params_)


# In[68]:


print('The best averaged performance:', grid_search.best_score_)


# In[69]:


model_best = grid_search.best_estimator_
print(f'The accuracy is: {model_best.score(X_test, Y_test)*100:.1f}%')


# In[70]:


pred = model_best.predict(X_test)


# In[72]:


print(classification_report(Y_test, pred, target_names=face_data.target_names))


# In[ ]:




