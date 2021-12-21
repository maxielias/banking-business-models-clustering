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
name = 'predict_data.csv'
predict_data = pd.read_csv(path+name)
predict_data = predict_data.drop(['Unnamed: 0'], axis=1)

print(predict_data['success'].sum())
print(len(predict_data['body']) - predict_data['success'].sum())
print(predict_data.loc[(predict_data['success'] == 0) & (predict_data['predict_cat'] == 'deportes')])

'''path = 'TP Final Eant/data/'
name = 'pp_data.csv'

pp_data = pd.read_csv(path+name)
print(pp_data.columns)
pp_data = pp_data.drop(['Unnamed: 0'], axis=1)
print(pp_data.columns)
pp_data.to_csv('TP Final Eant/data/pp_data.csv', index=False)
print(predict_data.columns)
#predict_data = predict_data.drop(['Unnamed: 0'], axis=1)
print(predict_data.columns)
predict_data.to_csv('TP Final Eant/data/predict_data.csv', index=False)'''
