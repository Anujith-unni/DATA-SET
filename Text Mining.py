#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary library
import pandas as pd
import numpy as np
import nltk
import os
import nltk.corpus
#sample text for performing tokenization
text='In Brazil they drive on the right-hand side of the road. Brazil has a large coastline on the eastern side of south america'
# importing word_tokenize from nlt
from nltk.tokenize import word_tokenize
#passing the string text into word tokenize for breaking the sentences
token=word_tokenize(text)
token


# In[2]:


#finding the frequency distinct in the tokens
#importing freDist library from nltk and assing token into FreqDist
from nltk.probability import FreqDist
fdist=FreqDist(token)
fdist


# In[3]:


# to find the frequency of top 10 words
fdist1=fdist.most_common(10)
fdist1


# In[5]:


#importing Porterstemmer from nltk library
#checking for the word 'giving'
from nltk.stem import PorterStemmer
pst=PorterStemmer()
pst.stem('waiting')


# In[6]:


#cheking for the list of words
stm=['waited',"waiting","waits"]
for word in stm: print(word+":"+pst.stem(word))


# In[7]:


#importing LancasterStemmer from nltk
from nltk.stem import LancasterStemmer
lst=LancasterStemmer()
stm=["giving",'given','given','gave']
for word in stm:print(word+':'+lst.stem(word))


# In[8]:


#importing Lemmatizer library from nltk
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

print('rocks :', lemmatizer.lemmatize('rocks'))
print('corpora :',lemmatizer.lemmatize('corpora'))


# In[9]:


#importing stopwords from nltk library
from nltk import word_tokenize
from nltk.corpus import stopwords
a=set(stopwords.words('english'))
text="Cristiano Ronaldo was born on Februray 5, 1985, in Funchal, Madeira,Portugal."
text1=word_tokenize(text.lower())
print(text1)
stopwords=[x for x in text1 if x not in a]
print(stopwords)


# In[10]:


text='vote to choose a particular man or a group (party) to represent them in parliament'
#tokenize the text
tex=word_tokenize(text)
for token in tex:
    print(nltk.pos_tag([token]))


# In[11]:


text="Google's CEO Sundar Pichai introduced the new Pixel at Minnesota Roi Centre Event"
#importing chunk library from nltk
from nltk import ne_chunk
#tokenize and POS tagging before the chunk 
token=word_tokenize(text)
tags=nltk.pos_tag(token)
chunk=ne_chunk(tags)
chunk


# In[13]:


text = 'We saw the yellow dog'
token=word_tokenize(text)
tags=nltk.pos_tag(token)

reg='NP: {<DT>?<JJ>*<NN>}'
a=nltk.RegexpParser(reg)
result=a.parse(tags)
print(result)


# In[ ]:




