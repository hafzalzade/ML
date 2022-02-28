#!/usr/bin/env python
# coding: utf-8

# In[232]:


import pandas as pd
data_raw = pd.read_csv('varedat4.csv')
print(data_raw.round(decimals=3).head(5))


# In[35]:


X = data_raw.drop('usd/w', axis=1).values
Y = data_raw['usd/w'].values


# In[108]:


X = data_raw.drop(['usd/w','ageent','own'], axis=1).values
Y = data_raw['usd/w'].values


# In[110]:


X = data_raw.drop(['usd/w','ageent'], axis=1).values
Y = data_raw['usd/w'].values


# In[125]:


X = data_raw.drop(['usd/w','own'], axis=1).values
Y = data_raw['usd/w'].values


# In[141]:


X = data_raw.drop(['usd/w','ageent','sel'], axis=1).values
Y = data_raw['usd/w'].values


# In[158]:


X = data_raw.drop(['usd/w','own','sel'], axis=1).values
Y = data_raw['usd/w'].values


# In[173]:


X = data_raw.drop(['usd/w','ageent','sel','own'], axis=1).values
Y = data_raw['usd/w'].values


# In[188]:


X = data_raw.drop(['usd/w','HS'], axis=1).values
Y = data_raw['usd/w'].values


# In[203]:


X = data_raw.drop(['usd/w','HS','sel'], axis=1).values
Y = data_raw['usd/w'].values


# In[218]:


X = data_raw.drop(['usd/w','HS','own'], axis=1).values
Y = data_raw['usd/w'].values


# In[233]:


X = data_raw.drop(['usd/w','HS','ageent'], axis=1).values
Y = data_raw['usd/w'].values


# In[142]:


print(X)
print(Y)


# In[5]:


print(X.shape)


# In[234]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# In[80]:


n_train = int(7030 * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]


# In[235]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)


# In[236]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.linear_model import SGDRegressor
param_grid = {"alpha": [1e-4, 3e-4, 1e-3],"eta0": [0.01, 0.03, 0.1],}
lr = SGDRegressor(penalty='l2', max_iter=1000, random_state=42)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='r2')
grid_search.fit(X_scaled_train, Y_train)


# In[237]:


print(grid_search.best_params_)
lr_best = grid_search.best_estimator_
predictions_lr = lr_best.predict(X_scaled_test)


# In[238]:


from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

print(f'MSE: {mean_squared_error(Y_test, predictions_lr):.3f}')
print(f'MAE: {mean_absolute_error(Y_test, predictions_lr):.3f}')
print(f'R^2: {r2_score(Y_test, predictions_lr):.3f}')


# In[239]:


from sklearn.ensemble import RandomForestRegressor
param_grid = {'max_depth': [30, 50],'min_samples_split': [2, 5, 10],'min_samples_leaf': [3, 5]}
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1,max_features='auto', random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, Y_train)


# In[240]:


print(grid_search.best_params_)
rf_best = grid_search.best_estimator_
predictions_rf = rf_best.predict(X_test)


# In[241]:


print(f'MSE: {mean_squared_error(Y_test, predictions_rf):.3f}')
print(f'MAE: {mean_absolute_error(Y_test, predictions_rf):.3f}')
print(f'R^2: {r2_score(Y_test, predictions_rf):.3f}')


# In[242]:


from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense 
model = Sequential([
        Dense(units=35, activation='relu'),
        Dense(units=1)
    ])


# In[243]:


import tensorflow as tf
model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.1))


# In[244]:


model.fit(X_scaled_train, Y_train, epochs=30, verbose=True)


# In[245]:


predictions = model.predict(X_scaled_test)


# In[246]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print(f'MSE: {mean_squared_error(Y_test, predictions):.3f}')
print(f'MAE: {mean_absolute_error(Y_test, predictions):.3f}')
print(f'R^2: {r2_score(Y_test, predictions):.3f}')


# In[ ]:




