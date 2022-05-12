#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
train_ds=pd.read_csv("sentiment_train",delimiter='\t')
train_ds.head(5)


# In[3]:


pd.set_option('max_colwidth',800)
train_ds[train_ds.sentiment==1][0:5]


# In[4]:


train_ds[train_ds.sentiment==0][0:5]


# In[5]:


train_ds.info()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(6,5))
#create count plot
ax=sn.countplot(x='sentiment',data=train_ds)
#annotate
for p in ax.patches:
    ax.annotate(p.get_height(),(p.get_x()+0.1,p.get_height()+50))


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer 
# Initialize the CountVectorizer
count_vectorizer = CountVectorizer() 
# Create the dictionary from the corpus
feature_vector = count_vectorizer.fit( train_ds.text ) 
# Get the feature names
features = feature_vector.get_feature_names() 
print( "Total number of features: ", len(features))


# In[11]:


import random
random.sample(features,10)


# In[12]:


train_ds_features=count_vectorizer.transform((train_ds.text))
type(train_ds_features)


# In[13]:


train_ds_features.shape


# In[14]:


train_ds_features.getnnz()


# In[15]:


print("Density of the matrix:",
     train_ds_features.getnnz()*100/
     (train_ds_features.shape[0]*train_ds_features.shape[1]))


# In[16]:


#Conerting the matriz to a dataframe
train_ds_df=pd.DataFrame(train_ds_features.todense())
#setting the column names to the features i.e. words
train_ds_df.columns=features


# In[17]:


train_ds_df.iloc[0:1,150:157]


# In[19]:


train_ds_df[['the','da','vinci','code','book','is','just','awesome']][0:1]


# In[20]:


#summing up the occurances of features column wise
features_counts=np.sum(train_ds_features.toarray(),axis=0)
feature_counts_df=pd.DataFrame(dict(features=features,
                                   counts=features_counts))


# In[21]:


plt.figure(figsize=(12,5))
plt.hist(feature_counts_df.counts,bins=50,range=(0,2000));
plt.xlabel('Frequency of words')
plt.ylabel('Density');


# In[22]:


len(feature_counts_df[feature_counts_df.counts==1])


# In[26]:


# Initialize the CountVectorizer
count_vectorizer = CountVectorizer(max_features=1000) 
# Create the dictionary from the corpus
feature_vector = count_vectorizer.fit( train_ds.text ) 
# Get the feature names
features = feature_vector.get_feature_names() 
# Transform the document into vectors
train_ds_features = count_vectorizer.transform( train_ds.text ) 
# Count the frequency of the features
features_counts = np.sum( train_ds_features.toarray(), axis = 0 ) 
feature_counts = pd.DataFrame( dict( features = features, 
counts = features_counts))


# In[27]:


feature_counts.sort_values('counts',ascending=False)[0:15]


# In[28]:


from sklearn.feature_extraction import text
my_stop_words=text.ENGLISH_STOP_WORDS
#Printing first few stop words
print("Few stop words:",list(my_stop_words)[0:10])


# In[29]:


#Adding custom words to the list of stop words
my_stop_words=text.ENGLISH_STOP_WORDS.union(['harry','potter','code','vinci','da','harry','mountain','movie','movies'])


# In[34]:


#setting stop words list
count_vectorizer=CountVectorizer(stop_words=my_stop_words,
                                 max_features=1000)
feature_vector=count_vectorizer.fit(train_ds.text)
train_ds_features=count_vectorizer.transform(train_ds.text)
features=feature_vector.get_feature_names()
features_counts=np.sum(train_ds_features.toarray(),axis=0)
feature_counts=pd.DataFrame(dict(features=features,
                                counts=features_counts))


# In[35]:


feature_counts.sort_values("counts",ascending=False)[0:15]


# In[ ]:




