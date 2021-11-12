#!/usr/bin/env python
# coding: utf-8

# In[160]:


from sklearn.datasets import fetch_lfw_people
face_data = fetch_lfw_people(min_faces_per_person=55)


# In[164]:


X = face_data.data
Y = face_data.target
z=face_data.target_names
print('Input data size :', X.shape)
print('Output data size :', Y.shape)


# In[165]:


print('Label names:', z)


# In[167]:


for i in range(9):
    print(f'Class {i} name {z[i]} has {(Y == i).sum()} samples.')


# In[222]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 4)
for i, axi in enumerate(ax.flat):
    axi.imshow(face_data.images[i], cmap='turbo')
    axi.set(xticks=[], yticks=[],xlabel=face_data.target_names[face_data.target[i]])


# In[223]:



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    random_state=42)


# In[224]:


from sklearn.svm import SVC
clf = SVC(class_weight='balanced', random_state=42)


# In[225]:


parameters = {'C': [0.1, 1, 8, 10],
              'gamma': [1e-07, 1e-08, 1e-06],
              'kernel' : ['rbf', 'linear','poly'] }


# In[226]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)


# In[227]:


grid_search.fit(X_train, Y_train)


# In[228]:


print('The best model:\n', grid_search.best_params_) 


# In[229]:


print('The best averaged performance:', grid_search.best_score_)


# In[230]:


clf_best = grid_search.best_estimator_

pred = clf_best.predict(X_test)


# In[231]:


print ('clf_best:\n', clf_best)
print('pred :\n' ,pred)


# In[232]:


from sklearn.metrics import classification_report
print(f'The accuracy is: {clf_best.score(X_test, Y_test)*100:.1f}%')


# In[233]:


print(classification_report(Y_test, pred,
                            target_names=face_data.target_names))


# In[234]:


from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=42)
svc = SVC(class_weight='balanced', kernel='rbf',random_state=42)
from sklearn.pipeline import Pipeline
model = Pipeline([('pca', pca),
                  ('svc', svc)])


# In[235]:


parameters_pipeline = {'svc__C': [1, 3, 10],
                       'svc__gamma': [0.001, 0.005]}
grid_search = GridSearchCV(model, parameters_pipeline)
grid_search.fit(X_train, Y_train)


# In[236]:


print('The best model:\n', grid_search.best_params_)


# In[237]:


print('The best averaged performance:', grid_search.best_score_)


# In[238]:


model_best = grid_search.best_estimator_
print(f'The accuracy is: {model_best.score(X_test, Y_test)*100:.1f}%')


# In[239]:


pred = model_best.predict(X_test)


# In[240]:


print(classification_report(Y_test, pred, target_names=face_data.target_names))


# In[ ]:




