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

### Abro el archivo 'linktitle' ###
path = ''
name = 'base.csv'
dflink = pd.read_csv(path+name)
#dflink = dflink.drop(['Unnamed: 0'], axis=1)
#dflink = dflink[]
#dflink = dflink.drop(['titulo', 'tit_stpwd', 'link'], axis=1)
print(dflink)

### Limpieza de los datos ###
import re
import string

def norm_text(text):
  #text = text.lower()  
  # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
  #text=re.sub('\s\W',' ',text)
  #text=re.sub('\W\s',' ',text)
  #text = text.replace('"', " ")
  # elimino palabras con 2 letras
  #text=re.sub(r'\W*\b\w{1,2}\b', '',text)
  # elimino los espacios dobles
  #text = re.sub('\n', ' ', text)
  #text=re.sub('\s+',' ',text)
  #text=re.sub('[\s{2,}]', ' ', text)
  #text = re.sub('^[ \t]+|[ \t]+$', '', text)
  #text = re.sub('[\t\r\n{2,}]', ' ', text)
  #" ".join(text.split())
  #text = text.replace('\t', '')
  #text = text.replace('\v', '')
  #text = text.replace('\n\r', '')
  #text = text.replace('\s', '')
  #text = re.sub(r'\s{2,}', ' ', text)
  #text = text.strip()
  #text=[sentence.replace('\r\n\r\n', '') for sentence in text]
  #text=[sentence.replace('\n', '') for sentence in text]
  #text=[sentence.replace('\r', ' ') for sentence in text]
  #text=[sentence.replace('\xa0', ' ') for sentence in text]
  #text=' '.join(text)
  #text = ' '.join(text.split())
  return text


def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?¿\]\%', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
 
# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…«»]', '', text)
    #text = re.sub('[\n]', '', text)
    return text

'''##### TITULO ########
round1 = lambda x: clean_text_round1(x)
#round1 = lambda x: norm_text(x)
 
dflink['final_tit'] = dflink['titulo'].apply(round1)
 
round2 = lambda x: clean_text_round2(x)
#round2 = lambda x: clean_text_round1(x)
 
dflink['final_tit'] = dflink['final_tit'].apply(round2)

round3 = lambda x: norm_text(x)
#round3 = lambda x: clean_text_round2(x)

dflink['final_tit'] = dflink['final_tit'].apply(round3)

#dflink['repr'] = [repr(titulo) for titulo in dflink['titulo']]

stpwrd_es = list(set(stopwords.words('spanish')))
round_stpwrd = lambda x: " ".join(x for x in x.split() if x not in stpwrd_es)
dflink['final_tit'] = dflink['final_tit'].apply(round_stpwrd)

#data_clean = dflink['final_tit']'''

##### CUERPO ########
round1 = lambda x: clean_text_round1(x)
#round1 = lambda x: norm_text(x)
 
dflink['final_cuerpo'] = dflink['cuerpo'].apply(round1)
 
round2 = lambda x: clean_text_round2(x)
#round2 = lambda x: clean_text_round1(x)
 
dflink['final_cuerpo'] = dflink['final_cuerpo'].apply(round2)

round3 = lambda x: norm_text(x)
#round3 = lambda x: clean_text_round2(x)

dflink['final_cuerpo'] = dflink['final_cuerpo'].apply(round3)

#dflink['repr'] = [repr(titulo) for titulo in dflink['titulo']]

stpwrd_es = list(set(stopwords.words('spanish')))
round_stpwrd = lambda x: " ".join(x for x in x.split() if x not in stpwrd_es)
dflink['final_cuerpo'] = dflink['final_cuerpo'].apply(round_stpwrd)

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
x = vectorizer.fit_transform(dflink['final_cuerpo'])

encoder = LabelEncoder()
y = encoder.fit_transform(dflink['categoria'])

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
voc_freq.to_csv('voc_freq.csv')
print(voc_freq)

#print(dflink['round3'][dflink['round3'].str.contains("ñba")])
#print(dflink['titulo'][48220])
#print(dflink['repr'][0:3])

'''def predict_cat(title):
    cod=nb.predict(vectorizer.transform([title]))
    return encoder.inverse_transform(cod)[0]

dflink['predict_cat'] = [predict_cat(titulo) for titulo in dflink['titulo']]
dflink['acierto'] = np.where(dflink['categoria'] == dflink['predict_cat'],1,0)
print(dflink['acierto'].sum())
print(len(dflink['titulo']) - dflink['acierto'].sum())

#no correr a menos que sea necesario
#retoques a los titulos
dflinkbody['tit_stpwd']=[norm_text(text) for text in dflinkbody['titulo']]
dflinkbody['tit_stpwd']=dflinkbody['tit_stpwd'].apply(lambda x: " ".join(x for x in x.split() if x not in stpwd_es))
dflinkbody['tit_stpwd']=dflinkbody['tit_stpwd'].apply(lambda x: " ".join(x for x in x.split() if x not in stpwdv1))
#retoques a los cuerpos de las notas
dflinkbody['cuerpo_stpwd']=[norm_text(text) for text in dflinkbody['body']]
dflinkbody['cuerpo_stpwd']=dflinkbody['cuerpo_stpwd'].apply(lambda x: " ".join(x for x in x.split() if x not in stpwd_es))
dflinkbody['cuerpo_stpwd']=dflinkbody['cuerpo_stpwd'].apply(lambda x: " ".join(x for x in x.split() if x not in stpwdv1))'''
