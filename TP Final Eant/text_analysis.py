# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:34:31 2021

@author: maxim
"""

# pip3 install -U scikit-learn scipy matplotlib --user
### Librerias a utilizar ###
import pandas as pd
import numpy as np

import nltk
#nltk.download('stopwords')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import spacy
spacy.load('en_core_web_trf')
spacy.load('es_dep_news_trf')

### Abro el archivo 'linktitle' ###
path = 'TP Final Eant/data/'
name = 'pp_data.csv'
pp_data = pd.read_csv(path+name)
pp_data = pp_data.drop(['Unnamed: 0'], axis=1)
'''print(pp_data['body'][11507:11508])
print(pp_data['pp_body'][11507:11508])
sub = 'þorsteinsdóttir'
print(pp_data[pp_data['pp_body'].str.find(sub) > -1])

vectordf = pd.DataFrame(pp_data['pp_body'].values.astype('U'))
vectordf.to_csv('TP Final Eant/data/vectordf.csv')'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(pp_data['pp_body'].values.astype('U'))

encoder = LabelEncoder()
y = encoder.fit_transform(pp_data['category'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# take a look at the shape of each of these
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

nb = MultinomialNB()
nb.fit(x_train, y_train)
print(nb.score(x_test, y_test))

x_test_pred = nb.predict(x_test)
confusion_matrix(y_test, x_test_pred)

print(classification_report(y_test, x_test_pred, target_names=encoder.classes_))

def make_reverse_vocabulary(vectorizer):
    revvoc = {}

    vocab = vectorizer.vocabulary_
    for w in vocab:
        i = vocab[w]

        revvoc[i] = w

    return revvoc

make_reverse_vocabulary(vectorizer)

voc_freq=make_reverse_vocabulary(vectorizer)
voc_freq=pd.DataFrame(voc_freq.items())
voc_freq.columns=['freq','word'] 
voc_freq=voc_freq.sort_values('freq',ascending=False)
voc_freq.to_csv('TP Final Eant/data/voc_freq.csv')
vectordf = pd.DataFrame(pp_data['pp_body'].values.astype('U'))
vectordf.to_csv('TP Final Eant/data/vectordf.csv')
#print(voc_freq)
#print(voc_freq)

def predict_cat(body):
    cod=nb.predict(vectorizer.transform([body]))
    return encoder.inverse_transform(cod)[0]

pp_data['predict_cat'] = [predict_cat(body) for body in pp_data['pp_body'].values.astype('U')]
pp_data['success'] = np.where(pp_data['category'] == pp_data['predict_cat'],1,0)

print(pp_data['success'].sum())
print(len(pp_data['body']) - pp_data['success'].sum())

pp_data.to_csv('TP Final Eant/data/predict_data.csv', index=False) 