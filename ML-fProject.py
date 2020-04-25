#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd  
import numpy as np  
import os, re
import string
import nltk 
from nltk.corpus import stopwords

#Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


#Current working directory
print(os.getcwd())

#Loading data
train = pd.read_csv('labeledTrainData.tsv',delimiter = '\t')
test = pd.read_csv('testData.tsv',delimiter = '\t')

train.shape, test.shape


# In[5]:


train.head()


# In[6]:


train['review'][0]


# In[7]:


test.head()


# In[8]:


test['review'][0]


# In[9]:


print ("number of rows for sentiment 1: {}".format(len(train[train.sentiment == 1])))
print ( "number of rows for sentiment 0: {}".format(len(train[train.sentiment == 0])))


# In[10]:


train.groupby('sentiment').describe().transpose()


# In[11]:


#Creating a new col
train['length'] = train['review'].apply(len)
train.head()


# In[12]:


Data Visualization
train['length'].plot.hist(bins = 100)


# In[13]:


train.length.describe()


# In[14]:


train[train['length'] == 13708]['review'].iloc[0]


# In[15]:


train.hist(column='length', by='sentiment', bins=100,figsize=(12,4))


# In[16]:


# Text Preprocessing
from bs4 import BeautifulSoup

#Creating a function for cleaning of data
def clean_text(raw_text):
    # 1. remove HTML tags
    raw_text = BeautifulSoup(raw_text).get_text() 
    
    # 2. removing all non letters from text
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                           
    
    # 4. Create variable which contain set of stopwords
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop word & returning   
    return [w for w in words if not w in stops]


# In[17]:


#Cleaning review and also adding a new col as its len count of words
train['clean_review'] = train['review'].apply(clean_text)
train['length_clean_review'] = train['clean_review'].apply(len)
train.head()


# In[18]:


train.describe()


# In[19]:


#Checking the smallest review
print(train[train['length_clean_review'] == 4]['review'].iloc[0])
print('------After Cleaning------')
print(train[train['length_clean_review'] == 4]['clean_review'].iloc[0])


# In[20]:


#Plot wordcloud
word_cloud = WordCloud(width = 1000, height = 500, stopwords = STOPWORDS, background_color = 'red').generate(
                        ''.join(train['review']))

plt.figure(figsize = (15,8))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# In[21]:


from sklearn.feature_extraction.text import CountVectorizer


# In[22]:


bow_transform = CountVectorizer(analyzer=clean_text).fit(train['review'])  #bow = bag of word

# Print total number of vocab words
print(len(bow_transform.vocabulary_))


# In[23]:



review1 = train['review'][1]
print(review1)


# In[24]:


bow1 = bow_transform.transform([review1])
print(bow1)
print(bow1.shape)


# In[25]:


print(bow_transform.get_feature_names()[71821])
print(bow_transform.get_feature_names()[72911])


# In[26]:


#Creating bag of words for our review variable
review_bow = bow_transform.transform(train['review'])


# In[27]:


print('Shape of Sparse Matrix: ', review_bow.shape)
print('Amount of Non-Zero occurences: ', review_bow.nnz)


# In[28]:


sparsity = (100.0 * review_bow.nnz / (review_bow.shape[0] * review_bow.shape[1]))
print('sparsity: {}'.format(sparsity))


# In[31]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(review_bow)
tfidf1 = tfidf_transformer.transform(bow1)
print(tfidf1)


# In[32]:


print(tfidf_transformer.idf_[bow_transform.vocabulary_['war']])
print(tfidf_transformer.idf_[bow_transform.vocabulary_['book']])


# In[33]:


review_tfidf = tfidf_transformer.transform(review_bow)
print(review_tfidf.shape)


# In[34]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train['review'], train['sentiment'], test_size=0.22, random_state=101)

len(X_train), len(X_test), len(X_train) + len(X_test)


# In[35]:


# #### Result Function
from sklearn.metrics import classification_report
def pred(predicted,compare):
    cm = pd.crosstab(compare,predicted)
    TN = cm.iloc[0,0]
    FN = cm.iloc[1,0]
    TP = cm.iloc[1,1]
    FP = cm.iloc[0,1]
    print("CONFUSION MATRIX ------->> ")
    print(cm)
    print()
    
    ##check accuracy of model
    print('Classification paradox :------->>')
    print('Accuracy :- ', round(((TP+TN)*100)/(TP+TN+FP+FN),2))
    print()
    print('False Negative Rate :- ',round((FN*100)/(FN+TP),2))
    print()
    print('False Postive Rate :- ',round((FP*100)/(FP+TN),2))
    print()
    print(classification_report(compare,predicted))


# In[36]:



from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)),  
    ('tfidf', TfidfTransformer()), 
    ('classifier', LogisticRegression(random_state=101)), 
])

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_train)
pred(predictions,y_train)


# In[37]:


#Test Set Result
predictions = pipeline.predict(X_test)
pred(predictions,y_test)


# In[38]:


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', MultinomialNB()), 
])

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_train)
pred(predictions,y_train)


# In[44]:


#Result on Test Case
predictions = pipeline.predict(X_test)
pred(predictions,y_test)


# In[43]:


from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)),  
    ('tfidf', TfidfTransformer()), 
    ('classifier', RandomForestClassifier(n_estimators = 500)), 
])

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_train)
pred(predictions,y_train)


# In[45]:


#Test Set Result
predictions = pipeline.predict(X_test)
pred(predictions,y_test)


# In[46]:


# Final Model Will be Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline_logit = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)), 
    ('tfidf', TfidfTransformer()),  
    ('classifier', LogisticRegression(random_state=101)),
])

pipeline_logit.fit(train['review'],train['sentiment'])
test['sentiment'] = pipeline_logit.predict(test['review'])


# In[47]:


test.head(5)


# In[48]:


output = test[['id','sentiment']]
print(output)


# In[49]:


output.to_csv( "output.csv", index=False, quoting=3 )

