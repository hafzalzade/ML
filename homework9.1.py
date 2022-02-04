#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download()


# In[2]:


from nltk.corpus import names

print(names.words()[:10])

print(len(names.words()))


# In[3]:


from nltk.tokenize import word_tokenize
sent = ''' من حمیدرضا افضل زاده هستم
          I'm learning Python Machine Learning By Dr. Rahmani,'''

print(word_tokenize(sent))


# In[4]:


sent2 = 'I am from I.R.A.N and have been to U.K. and U.S.A. , $20 in America , worked at P.T.K or Pars.Telephone.Kar P.T.Po p.t.kk .0.Pp. .Ka. Rr'
print(word_tokenize(sent2))


# In[5]:


import spacy

nlp = spacy.load('en_core_web_sm')
tokens2 = nlp(sent2)

print([token.text for token in tokens2])


# In[6]:


from nltk.tokenize import sent_tokenize
print(sent_tokenize(sent))


# In[7]:


import nltk
tokens = word_tokenize(sent)
print(nltk.pos_tag(tokens))


# In[8]:


nltk.help.upenn_tagset('PRP')
nltk.help.upenn_tagset('VBP')
nltk.help.upenn_tagset('JJ')
nltk.help.upenn_tagset('NNP')
nltk.help.upenn_tagset('CD')
nltk.help.upenn_tagset('DT')


# In[9]:


print([(token.text, token.pos_) for token in tokens2])


# In[10]:


tokens3 = nlp('The book written by Hayden Liu in 2020 was sold at $30 in america ,20 in IRAN')
print([(token_ent.text, token_ent.label_) for token_ent in tokens3.ents])


# In[11]:


from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
porter_stemmer.stem('machines')
porter_stemmer.stem('learning')


# In[12]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('machines')


# In[13]:


from sklearn.datasets import fetch_20newsgroups


groups = fetch_20newsgroups()
groups.keys()
groups['target_names']
groups.target


import numpy as np
np.unique(groups.target)



import seaborn as sns
sns.displot(groups.target)
import matplotlib.pyplot as plt
plt.show()


# In[14]:


groups.data[0]


# In[15]:


groups.target[0]


# In[16]:


groups.target_names[groups.target[0]]


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(max_features=500)
data_count = count_vector.fit_transform(groups.data)
data_count


# In[20]:


data_count[0]


# In[21]:


data_count.toarray()


# In[22]:


data_count.toarray()[0]


# In[23]:


print(count_vector.get_feature_names())


# In[24]:


from sklearn.feature_extraction import _stop_words
print(_stop_words.ENGLISH_STOP_WORDS)


# In[25]:


from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# In[27]:



all_names = set(names.words())


categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']
groups_3 = fetch_20newsgroups(categories=categories_3)

count_vector_sw = CountVectorizer(stop_words="english", max_features=500)

data_cleaned = []

for doc in groups_3.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)
    
data_cleaned_count_3 = count_vector_sw.fit_transform(data_cleaned)

tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)

data_tsne = tsne_model.fit_transform(data_cleaned_count_3.toarray())

plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_3.target)

plt.show()


# In[28]:


categories_5 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'comp.windows.x']
groups_5 = fetch_20newsgroups(categories=categories_5)

count_vector_sw = CountVectorizer(stop_words="english", max_features=500)

data_cleaned = []

for doc in groups_5.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)

data_cleaned_count_5 = count_vector_sw.fit_transform(data_cleaned)

data_tsne = tsne_model.fit_transform(data_cleaned_count_5.toarray())

plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_5.target)

plt.show()


# In[ ]:




