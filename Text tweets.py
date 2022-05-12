#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
tweets=pd.read_csv('Tweets.csv')
tweets.head()


# In[2]:


tweets_df=tweets.drop(tweets[tweets['airline_sentiment_confidence']<0.5].index,axis=0)
tweets_df.shape


# In[3]:


X=tweets_df['text']
y=tweets_df['airline_sentiment']


# In[4]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer


# In[5]:


stop_words=stopwords.words('english')
stemmer=PorterStemmer()


# In[6]:


import re
cleaned_data=[]
for i in range(len(X)):
    tweet=re.sub('[^a-zA-Z]',"",X.iloc[i])
    tweet=tweet.lower().split()


# In[7]:


tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words)]
tweet=''.join(tweet)
cleaned_data.append(tweet)


# In[8]:


cleaned_data


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000,stop_words=['virginamerica','unit'])
X_fin=cv.fit_transform(cleaned_data).toarray()


# In[10]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_fin,y,test_size=0.3)
model.fit(X_train,y_train)


# In[ ]:




