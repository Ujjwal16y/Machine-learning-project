#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[44]:


import nltk
nltk.download('stopwords')


# In[45]:


print(stopwords.words('english'))


# In[46]:


df=pd.read_csv('train.csv')


# In[47]:


df.head()


# In[48]:


df.shape


# In[49]:


df.isnull().sum()


# In[50]:


df=df.fillna('')


# In[51]:


df['content']=df['author']+' '+df['title']


# In[52]:


df['content']


# In[53]:


x=df.drop(columns='label',axis=1)
y=df['label']


# In[54]:


port_stem=PorterStemmer()


# In[55]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[56]:


df['content']=df['content'].apply(stemming)


# In[58]:


print(df['content'])


# In[59]:


x=df['content'].values
y=df['label'].values


# In[60]:


vectorizer=TfidfVectorizer()
vectorizer.fit(x)
x=vectorizer.transform(x)


# In[61]:


print(x)


# In[69]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[71]:


lr=LogisticRegression()
lr.fit(xtrain,ytrain)
pred=lr.predict(xtest)


# In[72]:


print("accuracy score on testing:",accuracy_score(ytest,pred))
print('accuracy score on training:-',(accuracy_score(lr.predict(xtrain),ytrain)))


# In[ ]:




