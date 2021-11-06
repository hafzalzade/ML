#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
X_train = np.array([
 [0, 1, 1],
 [0, 0, 1],
 [0, 0, 0],
 [1, 1, 0]])
Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[1, 1, 0]])
def get_label_indices(labels):
    from collections import defaultdict
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices
label_indices = get_label_indices(Y_train)
print('label_indices:\n', label_indices)
def get_prior(label_indices):
    prior = {label: len(indices) for label, indices in
                  label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior
prior = get_prior(label_indices)
print('Prior:', prior)


# In[62]:


def get_likelihood(features, label_indices, smoothing=0):
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood


# In[72]:


smoothing = 1
likelihood = get_likelihood(X_train, label_indices, smoothing)
print('Likelihood:\n', likelihood)


# In[105]:


def get_posterior(X, prior, likelihood):
    posteriors = []
    for x in X:
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
            sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


# In[106]:


posterior = get_posterior(X_test, prior, likelihood)
print('Posterior:\n', posterior)


# In[107]:


from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)


# In[108]:


pred_prob = clf.predict_proba(X_test)
print('[scikit-learn] Predicted probabilities:\n', pred_prob)


# In[109]:


pred = clf.predict(X_test)
print('[scikit-learn] Prediction:', pred)


# In[110]:


import numpy as np
from collections import defaultdict
data_path = 'ml-1m/ratings.dat'
n_users = 6040
n_movies = 3706


# In[122]:


def load_rating_data(data_path, n_users, n_movies):
    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split("::")
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping


# In[123]:


data, movie_n_rating, movie_id_mapping = load_rating_data(data_path, n_users, n_movies)


# In[124]:


def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of rating {int(value)}: {count}')
display_distribution(data)


# In[125]:


movie_id_most, n_rating_most = sorted(movie_n_rating.items(),
                                      key=lambda d: d[1], reverse=True)[0]
print(f'Movie ID {movie_id_most} has {n_rating_most} ratings.')


# In[126]:


X_raw = np.delete(data, movie_id_mapping[movie_id_most],axis=1)
Y_raw = data[:, movie_id_mapping[movie_id_most]]


# In[127]:


X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)


# In[128]:


display_distribution(Y)


# In[129]:


recommend = 3
Y[Y <= recommend] = 0
Y[Y > recommend] = 1
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f'{n_pos} positive samples and {n_neg} negative samples.')


# In[130]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=42)


# In[131]:


print(len(Y_train), len(Y_test))


# In[132]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)


# In[133]:


prediction_prob = clf.predict_proba(X_test)
print(prediction_prob[0:10])


# In[134]:


prediction = clf.predict(X_test)
print(prediction[:10])


# In[136]:


accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')


# In[ ]:




