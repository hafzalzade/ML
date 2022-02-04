#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
from sklearn.impute import SimpleImputer


data_origin = [[30, 100],
               [20, 50],
               [35, np.nan],
               [25, 80],
               [30, 70],
               [40, 60]]


imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(data_origin)
data_mean_imp = imp_mean.transform(data_origin)
print(data_mean_imp)


# In[22]:


imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(data_origin)
data_mean_imp = imp_mean.transform(data_origin)
print(data_mean_imp)


# In[23]:


imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median.fit(data_origin)
data_median_imp = imp_median.transform(data_origin)
print(data_median_imp)


# In[24]:


new = [[20, np.nan],
       [30, np.nan],
       [np.nan, 70],
       [np.nan, np.nan]]
new_mean_imp = imp_mean.transform(new)
print(new_mean_imp)


# In[25]:


from sklearn import datasets
dataset = datasets.load_diabetes()
X_full, y = dataset.data, dataset.target



m, n = X_full.shape
m_missing = int(m * 0.25)
print(m, m_missing)


# In[26]:


np.random.seed(42)
missing_samples = np.array([True] * m_missing + [False] * (m - m_missing))
np.random.shuffle(missing_samples)


missing_features = np.random.randint(low=0, high=n, size=m_missing)

X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = np.nan


# In[27]:


X_rm_missing = X_missing[~missing_samples, :]
y_rm_missing = y[~missing_samples]


# In[28]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=100)
score_rm_missing = cross_val_score(regressor, X_rm_missing, y_rm_missing).mean()
print(f'Score with the data set with missing samples removed: {score_rm_missing:.2f}')


# In[29]:


imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_mean_imp = imp_mean.fit_transform(X_missing)


# In[30]:


regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=100)
score_mean_imp = cross_val_score(regressor, X_mean_imp, y).mean()
print(f'Score with the data set with missing values replaced by mean: {score_mean_imp:.2f}')


# In[31]:


regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=500)
score_full = cross_val_score(regressor, X_full, y).mean()
print(f'Score with the full data set: {score_full:.2f}')


# In[32]:


import numpy as np
from sklearn.datasets import load_digits
dataset = load_digits()
X, y = dataset.data, dataset.target
print(X.shape)


# In[33]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
classifier = SVC(gamma=0.005, random_state=42)
score = cross_val_score(classifier, X, y).mean()
print(f'Score with the original data set: {score:.2f}')


# In[34]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
classifier = SVC(gamma=0.005, random_state=42)
score = cross_val_score(classifier, X, y).mean()
print(f'Score with the original data set: {score:.2f}')


# In[35]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1, random_state=42)
random_forest.fit(X, y)


# In[36]:


feature_sorted = np.argsort(random_forest.feature_importances_)


# In[37]:


K = [10, 15, 25, 35, 45,65,85]
for k in K:
    top_K_features = feature_sorted[-k:]
    X_k_selected = X[:, top_K_features]
    # Estimate accuracy on the data set with k selected features
    classifier = SVC(gamma=0.005)
    score_k_features = cross_val_score(classifier, X_k_selected, y).mean()
    print(f'Score with the dataset of top {k} features: {score_k_features:.2f}')


# In[38]:


from sklearn.datasets import load_digits
dataset = load_digits()
X, y = dataset.data, dataset.target

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


from sklearn.decomposition import PCA

# Keep different number of top components
N = [10, 15, 25, 35, 45,50,55]
for n in N:
    pca = PCA(n_components=n)
    X_n_kept = pca.fit_transform(X)
    # Estimate accuracy on the data set with top n components
    classifier = SVC(gamma=0.005)
    score_n_components = cross_val_score(classifier, X_n_kept, y).mean()
    print(f'Score with the dataset of top {n} components: {score_n_components:.2f}')


# In[39]:


from sklearn.preprocessing import Binarizer

X = [[4], [1], [3], [0]]
binarizer = Binarizer(threshold=2.9)
X_new = binarizer.fit_transform(X)
print(X_new)


# In[40]:


from sklearn.preprocessing import PolynomialFeatures

X = [[2, 4],
     [1, 3],
     [3, 2],
     [0, 3]]
poly = PolynomialFeatures(degree=2)
X_new = poly.fit_transform(X)
print(X_new)


# In[41]:


import gensim.downloader as api
model = api.load("glove-twitter-25")


# In[42]:


vector = model.most_similar('computer')
print('Word computer is embedded into:\n', vector)


# In[43]:


similar_words = model.most_similar("computer")
print('Top ten words most contextually relevant to computer:\n', similar_words)

