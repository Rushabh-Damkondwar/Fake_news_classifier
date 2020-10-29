#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('C:/Users/Rushabh/Downloads/Data science software/Python projects/Fake_news_classification_NLP/train.csv')


# In[3]:


df.head()


# In[4]:


df=df.dropna()


# In[5]:


## Get the Independent Features

X=df.drop('label',axis=1)


# In[6]:


X.head()


# In[7]:


## Get the Dependent features
y=df['label']


# In[8]:



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[9]:


df.isnull().sum()


# In[10]:


df = df.dropna()


# In[11]:


# To reset the index number to match the row numbers.

messages=df.copy()
messages.reset_index(inplace=True)
messages.head(10)


# In[12]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


# In[13]:


ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[14]:


# TFidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()


# In[15]:


X.shape


# In[16]:


y=messages['label']


# In[17]:


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[18]:


tfidf_v.get_feature_names()[:20]


# In[19]:


tfidf_v.get_params()


# In[20]:


count_df = pd.DataFrame(X_train, columns=tfidf_v.get_feature_names())


# In[21]:


count_df.head()


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[24]:


plt.show()


# In[25]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[26]:


from sklearn import metrics
import numpy as np
import itertools


# In[27]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[26]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score


# In[ ]:





# In[ ]:





# In[27]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[28]:


linear_clf = PassiveAggressiveClassifier()
linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.4f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])


# In[14]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

voc_size= 5000

onehot_repr=[one_hot(words,voc_size)for words in corpus] 


# In[15]:


onehot_repr[0]


# In[16]:



sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# In[17]:


## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[18]:


len(embedded_docs),y.shape


# In[19]:


y.head()


# In[20]:



import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


# In[ ]:





# In[21]:


X_final.shape,y_final.shape


# In[24]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=10)


# In[25]:


### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=50)


# In[26]:


#Adding dropout

from tensorflow.keras.layers import Dropout
## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[27]:


y_pred=model.predict_classes(X_test)


# In[28]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




