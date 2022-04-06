#####################################
# IMPORTO LIBRERIAS
# LIBRARIES
import pandas as pd
import numpy as np
#####################################
# CARGO EL ARCHIVO (DEBE ESTAR GUARDADO COMO TXT CON ";" COMO SEPARADOR, EL DECIMAL ES EL ".", Y SIN COMILLAS COMO SEPARADOR DE TEXTO)
# LOADING CSV FILE, RENAMIGN COLUMNS
evol_ind = pd.read_csv('data\evol_ind.csv', sep = ";", 
                         dtype = {'EENTIDAD': str, 'CODIND': str, 'FECHA': str, 'IND': str})
evol_ind = evol_ind.rename(columns={'EENTIDAD':'ent', 'CODIND':'cod', 'FECHA':'fecha', 'IND':'ind'})
evol_ind['ind'] = [ind.replace(',','.') for ind in evol_ind['ind']]
print(evol_ind)
"""# SACARLE LOS GH Y SISTEMA
# DELETING GROUPS KEEPING INDIVIDUAL BANKS FOR ANALYSIS
list_gh = ['1', '2', '3', '4', '5', '6', '7', '8', '11111', '00', '10', '11',
		'44099', '44098', '44096', '44095', '44088', '44093', '44092', '44090', '00340', '44059', '44094'
		]
evol_ind_loc= evol_ind[~evol_ind['ent'].isin(list_gh)]
# FILTRO POR INDICADORES A UTILIZAR
# FILTER NOT NEEDED INDICATORS
'''evol_ind = pd.read_csv('data/tit_ind.txt', sep = ",", dtype = {'CodInd': str, 'NomInd': str, 'Orden': int,
                                                                'Tipo': str, 'SForma': str, 'Formula': str,
                                                                'Alerta': str, 'Interpretación': str})'''
list_ind = [
          'R2', 'R1', 'E2', 'E1', 'RAPM',
          'A6', 'AG1', 'A2', 'A21',
          'L1', 'L8',
          'C1', 'C14', 'C2', 
          'LS1', 'LS2', 'MS1'
		]
evol_ind_loc= evol_ind_loc[evol_ind_loc['cod'].isin(list_ind)]
# PROMEDIO DOCE MESES EN NUEVA COLUMNA
# 48 ROLLING WINDOW AVERAGE
evol_ind_loc['avg_ind'] = evol_ind_loc['ind'].rolling(window=48).mean()
# ELIMINO FILAS CON VALOR NAN
# DROPPING NAN VALUES
evol_ind_loc = evol_ind_loc.dropna(axis=0)
# FILTRO POR UNA FECHA PARTICULAR
# FILTER BY DATE
evol_ind_loc = evol_ind_loc[evol_ind_loc['fecha'] == '31/12/2020 00:00:00']
# ELIMINO COLUMNA FECHA
# DROPPING DATE
evol_ind_loc = evol_ind_loc.drop(columns=['fecha'])
evol_ind_loc = evol_ind_loc.reset_index()
# print(evol_ind_loc.loc[evol_ind_loc['cod'].duplicated(keep=False)])
# print(evol_ind_loc[(evol_ind_loc['cod'] == 'A10')])
# CREO LA TABLA PIVOTE (COLUMNAS=INDICADORES, FILAS=ENTIDADES)
# CREATING PIVOT TABLE
evol_ind_piv = evol_ind_loc.pivot(index='ent', columns='cod', values='ind').reset_index()
evol_ind_piv = evol_ind_piv.set_index('ent')
# ELIMINO ENTIDADES QUE NO TENGAN ROA
list_ent = evol_ind_piv.index[evol_ind_piv['RG1II'].isna()].values
evol_ind_piv = evol_ind_piv[~evol_ind_piv.index.isin(list_ent)]
# SI ALGUNA ENTIDAD NO PRESENTA ALGUN INDICADOR LE PONGO VALOR = 0
evol_ind_piv = evol_ind_piv.fillna(0)
# print(evol_ind_piv.shape)
# SI LA MAYORÍA DE LOS VALORES DEL INDICADOR SON CERO, LO ELIMINO
# DROPPING COLUMNS WHERE THE NUMBER OF VALUES = 0 ARE 50% OR MORE
zeroscol = evol_ind_piv.astype(bool).sum(axis=0)/len(evol_ind_piv)
zeroscol_list = zeroscol[zeroscol < 0.5].index
for ind in zeroscol_list:
      if ind in evol_ind_piv.columns:
            del evol_ind_piv[ind]
# print(evol_ind_piv.shape)
print(evol_ind_piv)
#####################################
# AGREGO LOS GH
gh = pd.read_csv('data\gh.txt', sep = ",", dtype = {'CODIGO': str, 'ENTIDAD': str, 'Grupo': int})
gh = gh.set_index('CODIGO')
evol_ind_piv = pd.concat([evol_ind_piv, gh['Grupo']], axis=1, join='inner')
# print(evol_ind_piv.shape)
# print(evol_ind_piv)
#####################################
cor_matrix = evol_ind_piv.corr().abs()
# print(cor_matrix)
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
# print(upper_tri)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
# print(to_drop)
evol_ind_corr = evol_ind_piv.drop(columns=to_drop, axis=1)
# print(evol_ind_corr.shape)
#####################################
# IMPORTO LIBRERIAS
# IMPORTINF LIBRARIES SKLEARN AND PLOTTING
from pandas.plotting import scatter_matrix
from pandas import DataFrame
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
#####################################
# LOF
# data visualzation
import matplotlib.pyplot as plt
import seaborn as sns
# outlier/anomaly detection
from sklearn.neighbors import LocalOutlierFactor
'''
# model specification
model1 = LocalOutlierFactor(n_neighbors = 20, metric = "manhattan", contamination = 0.02)
# model fitting
y_pred = model1.fit_predict(evol_ind_corr)
# filter outlier index
outlier_index = np.where(y_pred == -1) # negative values are outliers and positives inliers
# filter outlier values
outlier_values = evol_ind_corr.iloc[outlier_index]'''
#####################################
# CREATE FUNCTION TO DETERMINE LOCAL OUTLIER FACTOR FOR OUTLIER DETECTION
def lof(data, predict=None, k=20, method=1, plot=False):
    
    # Determine si se pasan los datos de la prueba. De lo contrario, los datos de la prueba se asignan como datos de entrenamiento.
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    predict = pd.DataFrame(predict)
         # Calcular LOF atípico
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1)
    clf.fit(data)
         # Record k distancia del vecindario
    predict['k distances'] = clf.kneighbors(predict)[0].max(axis=1)
         # Grabe el valor atípico de LOF y haga lo contrario
    predict['local outlier factor'] = -clf._decision_function(predict.iloc[:, :-1])
         # Separe los puntos del clúster de los puntos normales según el umbral
    outliers = predict[predict['local outlier factor'] > method].sort_values(by='local outlier factor')
    inliers = predict[predict['local outlier factor'] <= method].sort_values(by='local outlier factor')
    return outliers, inliers
#####################################
# APPLYING LOF FUNCTION TO DROP OUTLIERS
# print(evol_ind_corr.head())
outlier_values, inlier_values = lof(evol_ind_corr)
# print(list_ent)
# print(list(outlier_values.index))
# print(list(inlier_values.index))
evol_ind_lof = evol_ind_corr[~evol_ind_corr.index.isin(outlier_values.index)]
#####################################
# DESCRIPCION DE LOS DATOS Y ESCALAR DATOS
# TRANSFORMO EN NUMPY ARRAY
# NORMALIZING VALUES WITH ROBUST SCALER
evol_ind_np = evol_ind_lof.to_numpy()
robustScaler = RobustScaler()
x = evol_ind_lof.drop(['Grupo'], axis=1) # evol_ind_np[:, :-1]
y = evol_ind_lof['Grupo'] # evol_ind_np[:, -1:]
data = robustScaler.fit_transform(x)
list_col = evol_ind_lof.columns
list_col = list_col[:-1]
data = DataFrame(data, columns=list_col)
# print(evol_ind_lof.shape)
# print(data.shape)
# print(data)
# print(x.shape)
# print(y.shape)
####################################
# REDUCING DIMENSIONALITY WITH PCA METHOD - KEEPING SIGNIFICATIVE VARIABLES
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
pcadf = pd.DataFrame(data = principalComponents
             , columns = ['pca1', 'pca2', 'pca3'])
# print(pcadf)
# pca_data = pd.concat([pcadf, y], axis=1)
# print(pca_data[~pca_data['Grupo'].isna()])
explained_variance_df = pca.explained_variance_ratio_
explained_variance_df = np.insert(explained_variance_df, 0, 0)
cumulative_variance = np.cumsum(np.round(explained_variance_df, decimals=3))
pca_variance = pd.DataFrame(['', 'pc1', 'pc2', 'pc3'], columns=['pc'])
explained_variance_df = pd.DataFrame(explained_variance_df, columns=['explained_variance'])
cumulative_variance = pd.DataFrame(cumulative_variance, columns=['cumulative_variance'])
explained_variance_df = pd.concat([pca_variance, explained_variance_df, cumulative_variance], axis=1)
print(explained_variance_df)
'''####################################
import plotly.express as px
fig = px.bar(explained_variance_df,
                  x='pc', y='explained_variance',
                  width=800)
fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.show()
# plt.scatter(evol_ind_corr.index, evol_ind_corr, color = "b", s = 65)
# plt.grid()
# plt.show()'''
####################################
# DIANA DIVISIVE ANALYSIS
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(evol_ind_piv, 'single')

labelList = range(1, 11)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=evol_ind_piv.index,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()"""