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
nltk.download('stopwords')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import spacy
spacy.load('en_core_web_trf')
spacy.load('es_dep_news_trf')

### Abrir 'raw_data' ###
path = 'TP Final Eant/data/'
name = 'raw_data.csv'
raw_data = pd.read_csv(path+name)
#print(raw_data)

### Limpieza de los datos ###
import re
import string
from text_to_num import alpha2digit

pp_data = raw_data

def norm_text(text): 
  '''1) Delete single characters,
     2) Delete brackets,
     3) Delete symbols,
     4) Text to number,
     5) Delete trailing whitespaces,
     6) Lower case
  '''
  text = re.sub(r'\b[a-zA-Z]\b', '', str(text))
  text = re.sub(r'[\([{})\]]', '', text)
  text = re.sub(r'(?<!\d)[.,;:()¿?!¡-](?!\d)', ' ', text)
  text = alpha2digit(text, 'es')
  text = " ".join(text.split())
  text = text.lower()
  return text

##### Limpio y proceso títulos ########
round1 = lambda x: norm_text(x)
pp_data['pp_title'] = pp_data['title'].apply(round1)

stpwrd_es = list(set(stopwords.words('spanish')))
round2 = lambda x: " ".join(x for x in x.split() if x not in stpwrd_es)
pp_data['pp_title'] = pp_data['pp_title'].apply(round2)

##### Limpio y proceso cuerpo de los artículos ########
round1 = lambda x: norm_text(x)
pp_data['pp_body'] = pp_data['body'].apply(round1)

stpwrd_es = list(set(stopwords.words('spanish')))
round2 = lambda x: " ".join(x for x in x.split() if x not in stpwrd_es)
pp_data['pp_body'] = pp_data['pp_body'].apply(round2)

pp_data.to_csv('TP Final Eant/data/pp_data.csv', index=False)