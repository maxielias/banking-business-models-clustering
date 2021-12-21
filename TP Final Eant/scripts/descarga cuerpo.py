# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:00:03 2020

@author: maxim
"""
import os
import pandas as pd
import re

#### defino ruta de trabajo donde tengo mis archivos csv
os.getcwd()
os.listdir()
os.chdir('C:/Users/maxim/Documents/TRABAJO FINAL')

#### abro el archivo
name='actualiza_titulo+link.csv'
dftitnew=pd.DataFrame(pd.read_csv(name))
dftitfull=pd.DataFrame(pd.read_csv(name))


#### descargo el cuerpo de cada link
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
from newspaper import fulltext

inter=[]
link=[]
cuerpo=[]
for l in dftitnew['link']:    
    try:
      html=requests.get(l).text
      text=fulltext(html)
      link.append(l)
      cuerpo.append(text)                
    except:
        pass

dicc={'link':link,'cuerpo':cuerpo}
dfcuerpo=pd.DataFrame(dicc)
dfcuerpo.to_csv(r'C:/Users/maxim/Documents/TRABAJO FINAL/cuerponew.csv')

#dftitfinal[dftitfinal.categoria=='deportes']
#dfcuerpo[dftitfinal.categoria=='deportes']

dfcuerpo

#---------------


name='cuerponew.csv'
dfcuerpo=pd.DataFrame(pd.read_csv(name))

dftitnew=dftit.merge(dfcuerpo[['link', 'cuerpo']], on=['link'])

import nltk
nltk.download('stopwords')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import spacy
import matplotlib.pyplot as plt

from pathlib import Path
#fileContent = Path(url).read_text()
sw = list(set(stopwords.words('spanish'))) ##### DEFINO UN DICCIONARIO DE STOP WORDS

name='stopwords1.doc'
f=open("stopwords1.txt", "r")
stopwords1 = list()
with f as f:
  for line in f:
    stopwords1.append(line)
  stopwords1 = [line.rstrip('\n') for line in stopwords1]

stopwords1

def norm_text(text):
  text = text.lower()
  
  # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
  text=re.sub('\s\W',' ',text)
  text=re.sub('\W\s',' ',text)

  # elimino palabras con 2 letras
  text=re.sub(r'\W*\b\w{1,2}\b', '',text)
  
  # make sure we didn't introduce any double spaces
  text=re.sub('\s+',' ',text)
  
  return text

dftitnew['tit_norm']=[norm_text(text) for text in dftitnew['titulo']]
dftitnew['tit_stpwd']=dftitnew['tit_norm'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
dftitnew['tit_stpwd']=dftitnew['tit_stpwd'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords1))

dftitnew['cuerpo_norm']=[norm_text(text) for text in dftitnew['cuerpo']]
dftitnew['cuerpo_stpwd']=dftitnew['cuerpo_norm'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
dftitnew['cuerpo_stpwd']=dftitnew['cuerpo_stpwd'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords1))

dftitnew
#dftitfinal.to_csv(r'C:/Users/maxim/Documents/TRABAJO FINAL/dftitfinal.csv')
#dftitfinal.to_excel(r'C:/Users/maxim/Documents/TRABAJO FINAL/dftitfinal.xlsx')

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
x = vectorizer.fit_transform(dftitfinal['cuerpo_stpwd'])

encoder = LabelEncoder()
y = encoder.fit_transform(dftitfinal['categoria'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

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

#voc_freq=make_reverse_vocabulary(vectorizer)
#voc_freq=pd.DataFrame(voc_freq.items())
#voc_freq.columns=['freq','word'] 
#voc_freq=voc_freq.sort_values('freq',ascending=False)
#voc_freq.to_csv('/content/drive/My Drive/PROGRAMACION/TRABAJO FINAL/voc_freq.csv')
#voc_freq

def predict_cat(title):
    cod=nb.predict(vectorizer.transform([title]))
    return encoder.inverse_transform(cod)[0]

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#Links que funcionan:"https://cronista.com/","https://ambito.com/","https://iprofesional.com/"

#Links que NO funcionan:
#"https://lanacion.com.ar/","https://página12.com.ar"


#Rellenar lista con diarios de interés:
diarios=["https://cronista.com/"]
#,"https://ambito.com/","https://iprofesional.com/"]

maestro=[]
for i in diarios:
  page = requests.get(i)
  page.content

  soup=BeautifulSoup(page.content,"html.parser")
  bs=soup.select("a[title]")  

  titulo=[]
  text=[]
  for i in range(len(bs)):
    bs1=bs[i].get_text()
    if bs1.count(" ")>4:
      titulo.append(bs1)
    else:
      continue
  
#### descargo el cuerpo de cada link
#import requests
#from bs4 import BeautifulSoup
#from urllib.request import urlopen
#from newspaper import fulltext

#inter=[]
#link=[]
#cuerpo=[]
#for l in dftit['link']:    
#    try:
#      html=requests.get(l).text
#      text=fulltext(html)

text={'titulo':text}
text=pd.DataFrame(text)
text['tit_norm']=[norm_text(text) for text in text['titulo']]
text['tit_stpwd']=text['tit_norm'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
text['tit_stpwd']=text['tit_stpwd'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords1))
text['tit_pred']=[predict_cat(text) for text in text['tit_stpwd']]

text.to_csv(r'C:/Users/maxim/Documents/TRABAJO FINAL/predict_titulo.csv')
text.to_excel(r'C:/Users/maxim/Documents/TRABAJO FINAL/predict_titulo.xlsx')

text

text.loc[(text['tit_pred']=='economia')]

from bs4 import BeautifulSoup
from urllib.request import urlopen
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

corpus = ' '.join(dftitfinal['cuerpo_stpwd'][dftitfinal['categoria']=='deportes'])
corpus
wordcloud = WordCloud(width = 1000, height = 500, background_color="skyblue", colormap="YlOrRd").generate(corpus)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("nube"+".png", bbox_inches='tight')
plt.show()
plt.close()


######--------------------------------------------------------------

inter=[]
link=[]
cuerpos=[]
for l in dftitnew['link'][0:1]:    
    try:
      with urlopen(l) as response:
        soup=BeautifulSoup(response,'html.parser')
        h2=soup.find_all('h2')
        p=soup.find_all('p')
#        for anchor in h2:            
#                link.append(anchor.a['href'])                
#                for anchor in p:
#                  inter.append(anchor.text)
#                inter=' '.join([str(item) for item in inter])
#                cuerpos.append(inter)
#                inter=[]                
    except:
        pass
print(h2)
print(p)

#________________________________________________________

from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import re
from newspaper import fulltext

link=[]
titulo=[]
categoria=[]
n=list(range(0,2))
topico=['https://buscar.lanacion.com.ar/economia/c-Econom%C3%ADa/page-',
        'https://buscar.lanacion.com.ar/politica/c-Pol%C3%ADtica/page-',
        'https://buscar.lanacion.com.ar/cultura/c-Cultura/page-',
        'https://buscar.lanacion.com.ar/personajes/c-Espect%C3%A1culos/page-',
        'https://buscar.lanacion.com.ar/sociedad/c-Sociedad/page-',
        'https://buscar.lanacion.com.ar/turismo/c-Turismo/page-',
        'https://buscar.lanacion.com.ar/deportes/c-Deportes/page-']

for i in n:
    try:
        for t in topico:
          with urlopen(t+str(i)) as response:
              soup=BeautifulSoup(response,'html.parser')
              h2=soup.find_all('h2')
              title=soup.find_all('title')      
              if h2.a['href']!="/":
                    titulo.append(h2.a.text.strip())
                    link.append(h2.a['href'])  
                    categoria.append(title.text.split('-')[0].strip())        

#              for anchor in h2:            
#                if anchor.a['href']!="/":
#                    titulo.append(anchor.a.text.strip())
#                    link.append(anchor.a['href'])                
#                    for anchor in title:                               
#                        categoria.append(anchor.text.split('-')[0].strip())        
    except:
        pass  

dicc={'link':link,'titulo':titulo,'categoria':categoria}
prueba=pd.DataFrame(dicc)

print(link)
print(titulo)
print(categoria)

len(link)
len(titulo)
len(categoria)

i
prueba