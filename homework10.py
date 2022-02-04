#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target


# In[2]:


import numpy as np
from matplotlib import pyplot as plt
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()


# In[33]:


k = 7
random_index = np.random.choice(range(len(X)), k)
centroids = X[random_index]


# In[34]:


def visualize_centroids(X, centroids):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
    plt.show()


# In[35]:


visualize_centroids(X, centroids)


# In[36]:


def dist(a, b):
    return np.linalg.norm(a - b, axis=1)


# In[37]:


def assign_cluster(x, centroids):
    distances = dist(x, centroids)
    cluster = np.argmin(distances)
    return cluster


# In[38]:


def update_centroids(X, centroids, clusters):
    for i in range(k):
        cluster_i = np.where(clusters == i)
        centroids[i] = np.mean(X[cluster_i], axis=0)


clusters = np.zeros(len(X))

tol = 0.0001
max_iter = 100    

iter = 0
centroids_diff = 100000


# In[39]:


from copy import deepcopy
while iter < max_iter and centroids_diff > tol:
    for i in range(len(X)):
        clusters[i] = assign_cluster(X[i], centroids)
    centroids_prev = deepcopy(centroids)
    update_centroids(X, centroids, clusters)
    iter += 1
    centroids_diff = np.linalg.norm(centroids - centroids_prev)
    print('Iteration:', str(iter))
    print('Centroids:\n', centroids)
    print('Centroids move: {:5.4f}'.format(centroids_diff))
    visualize_centroids(X, centroids)


# In[40]:


plt.scatter(X[:,0], X[:,1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*',s=200, c='#050505')
plt.show()


# In[43]:


from sklearn.cluster import KMeans
kmeans_sk = KMeans(n_clusters=5, random_state=42)
kmeans_sk.fit(X)
clusters_sk = kmeans_sk.labels_
centroids_sk = kmeans_sk.cluster_centers_


plt.scatter(X[:, 0], X[:, 1], c=clusters_sk)
plt.scatter(centroids_sk[:, 0], centroids_sk[:, 1], marker='*', s=200, c='#050505')
plt.show()


# In[12]:


iris = datasets.load_iris() 
X = iris.data
y = iris.target
k_list = list(range(1, 7))
sse_list = [0] * len(k_list)


# In[13]:


for k_ind, k in enumerate(k_list):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_

    sse = 0
    for i in range(k):
        cluster_i = np.where(clusters == i)

        sse += np.linalg.norm(X[cluster_i] - centroids[i])

    print('k={}, SSE={}'.format(k, sse))
    sse_list[k_ind] = sse



plt.plot(k_list, sse_list)
plt.show()


# In[14]:


from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
    
]


# In[15]:


groups = fetch_20newsgroups(subset='all', categories=categories)
labels = groups.target
label_names = groups.target_names

from nltk.corpus import names
all_names = set(names.words())



from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_cleaned = []


# In[16]:


for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)

data = count_vector.fit_transform(data_cleaned)


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words="english",max_features=None, max_df=0.5, min_df=2)
data = count_vector.fit_transform(data_cleaned)


# In[18]:


from sklearn.cluster import KMeans
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)


# In[19]:


clusters = kmeans.labels_
from collections import Counter
print(Counter(clusters))


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vector = TfidfVectorizer(stop_words='english', max_features=None, max_df=0.5, min_df=2)


# In[21]:


data = tfidf_vector.fit_transform(data_cleaned)
kmeans.fit(data)
clusters = kmeans.labels_
print(Counter(clusters))


# In[22]:


import numpy as np
cluster_label = {i: labels[np.where(clusters == i)] for i in range(k)}
terms = tfidf_vector.get_feature_names()
centroids = kmeans.cluster_centers_
for cluster, index_list in cluster_label.items():
    counter = Counter(cluster_label[cluster])
    print('cluster_{}: {} samples'.format(cluster, len(index_list)))
    for label_index, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        print('{}: {} samples'.format(label_names[label_index], count))
    print('Top 10 terms:')
    for ind in centroids[cluster].argsort()[-10:]:
        print(' %s' % terms[ind], end="")
    print()


# In[23]:


from sklearn.decomposition import NMF
t = 20
nmf = NMF(n_components=t, random_state=42)


# In[24]:


data = count_vector.fit_transform(data_cleaned)


# In[25]:


nmf.fit(data)


# In[26]:


nmf.components_


# In[27]:


for topic_idx, topic in enumerate(nmf.components_):
        print("Topic {}:" .format(topic_idx))
        print(" ".join([terms[i] for i in topic.argsort()[-10:]]))


# In[28]:


from sklearn.decomposition import LatentDirichletAllocation
t = 20
lda = LatentDirichletAllocation(n_components=t,learning_method='batch',random_state=42)


# In[29]:


data = count_vector.fit_transform(data_cleaned)


# In[30]:


lda.fit(data)


# In[31]:


lda.components_


# In[32]:


terms = count_vector.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic {}:" .format(topic_idx))
    print(" ".join([terms[i] for i in
            topic.argsort()[-10:]]))


# In[ ]:




