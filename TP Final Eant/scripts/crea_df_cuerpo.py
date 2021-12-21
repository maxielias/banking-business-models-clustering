# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:32:43 2020

@author: maxim
"""

#Actualizo el archivo
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import re
import os

os.getcwd()
os.listdir()
os.chdir('C:/Users/maxim/Documents/TRABAJO FINAL')

#### abro el archivo
name1='linktitle.csv'
name2='body60k.csv'
df=pd.DataFrame(pd.read_csv(name1))
body=pd.DataFrame(pd.read_csv(name2))

del df['Unnamed: 0']
del body['Unnamed: 0']

########## ARMO DATAFRAME CON LOS LINKS DE LAS NOTAS PARA GENERAR EL MODELO ##########
link=[]
tit=[]
cat=[]
body=[]
n=list(range(1,1000))
topico=['https://buscar.lanacion.com.ar/economia/c-Econom%C3%ADa/page-','https://buscar.lanacion.com.ar/politica/c-Pol%C3%ADtica/page-',
        'https://buscar.lanacion.com.ar/cultura/c-Cultura/page-','https://buscar.lanacion.com.ar/personajes/c-Espect%C3%A1culos/page-',
        'https://buscar.lanacion.com.ar/sociedad/c-Sociedad/page-','https://buscar.lanacion.com.ar/turismo/c-Turismo/page-',
        'https://buscar.lanacion.com.ar/deportes/c-Deportes/page-']

for i in n:
    try:
        for t in topico:
          with urlopen(t+str(i)) as response:
              soup=BeautifulSoup(response,'html.parser')
              h2=soup.find_all('h2')
              title=soup.find_all('title')
              for h in h2:                
                if h.a['href']!="/":
                  tit.append(h.a.text.strip())
                  link.append(h.a['href'])  
                  for t in title:
                    cat.append(t.text.split('-')[0].strip())        
    except:
        pass  
    
dic={'link':link,'titulo':tit,'categoria':cat}
df=pd.DataFrame(dic)
df.to_csv(r'C:/Users/maxim/Documents/TRABAJO FINAL/linktitle.csv')

########## ARMO DATAFRAME CON LOS LINKS DE LAS NOTAS PARA GENERAR EL MODELO ##########
bodylist=[]
prueba=[]
for l in df['link'][60000:69930]:
    try:
        with urlopen(l) as response:
            soup=BeautifulSoup(response,'html.parser')
            p=soup.find_all('p')
            art=[tag.get_text().strip() for tag in p]
            art=[sentence for sentence in art if sentence.endswith('.')]
            art=[sentence for sentence in art if not sentence.startswith('Los comentarios publicados')]
            art=[sentence for sentence in art if not sentence.startswith('Descargá la aplicación')]
            art=[sentence.replace('\r\n\r\n', '') for sentence in art]
            art=[sentence.replace('\n', '') for sentence in art]
            art=[sentence.replace('\r', ' ') for sentence in art]
            art=' '.join(art)
            bodylist.append(art)
    except:
        bodylist.append(0)
        pass
dicbody={'body':bodylist}
bodynew=pd.DataFrame(dicbody)
body=pd.concat([body,bodynew])
body.to_csv(r'C:/Users/maxim/Documents/TRABAJO FINAL/body69k.csv')
##########
#p=soup.find_all('p')
#art=[tag.get_text().strip() for tag in p]
#art=[sentence for sentence in art if '\n' in sentence]
#art=[sentence for sentence in art if sentence.endswith('.')]
#art=['\r\n\r\n'.join(sentence.splitlines()) for sentence in art]
#art=['\r'.join(sentence.splitlines()) for sentence in art]
#art=[sentence.replace('\r\n\r\n', '') for sentence in art]
#art=[sentence.replace('\n', '') for sentence in art]
#art=[sentence.replace('\r', ' ') for sentence in art]
#art=' '.join(art)
#print(art)