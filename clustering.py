import pandas as pd
import numpy as np

evol_ind = pd.read_csv('data\evol_ind.csv', sep = ";", dtype = {'EENTIDAD': str, 'CODIND': str, 'FECHA': str, 'IND': str})
evol_ind = evol_ind.rename(columns={'EENTIDAD':'ent', 'CODIND':'cod', 'FECHA':'fecha', 'IND':'ind'})
evol_ind['ind'] = [float(ind.replace(',','.')) for ind in evol_ind['ind']]

entidad = pd.read_csv('data\ENTIDAD.csv', sep = ";", dtype = {
    'EENTIDAD': str,'CODNUM': str,'CODIGO SC': str,'ENOMBRE': str,'NOMBRE CORTO': str,'NOMBRE IG': str,
    'IG': str,'GRUPO ENCUESTA': str,'NOMBRE GRUPO': str,'GRUPO ENCUESTA Viejo': str,'SUBGRUPO': str,
    'SECTOR': str,'TIPO': str,'MC': str,'GH': str,'NGH': str,'JEFE GRUPO VIEJO': str,'CF': str,
    'CREFI': str,'JEFE GRUPO': str,'ANALISTA': str,'VIGENTE': str,'FECHA_ALTA': str,
    'BAJA': str,'LEGAJO': str,'SubJefe': str,'GRUPO': str
    }, encoding="cp1252")

list_gh = ['1', '2', '3', '4', '5', '6', '7', '8', '11111', '00', '10', '11',
		'44099', '44098', '44096', '44095', '44088', '44093', '44092', '44090', '00340', '44059', '44094'
		]
evol_ind_loc= evol_ind[~evol_ind['ent'].isin(list_gh)]

'''evol_ind = pd.read_csv('data/tit_ind.txt', sep = ",", dtype = {'CodInd': str, 'NomInd': str, 'Orden': int,
                                                                'Tipo': str, 'SForma': str, 'Formula': str,
                                                                'Alerta': str, 'Interpretación': str})'''

list_ind = [
          'RG2', 'RG1', 'E2', 'E1', # 'RAPM',
          'A6', 'AG1', 'A2', 'A21',
          'L1', 'L8',
          'C1', 'C14', 'C2', 
          'LS1', 'LS2', 'MS1'
		]
evol_ind_loc = evol_ind_loc[evol_ind_loc['cod'].isin(list_ind)]

ent_vig = entidad[['EENTIDAD','NOMBRE CORTO','SECTOR','TIPO','MC','GH','NGH','VIGENTE','FECHA_ALTA',
                    'BAJA','GRUPO']]
ent_vig = ent_vig[ent_vig['VIGENTE']=='VERDADERO']

evol_ind_loc = evol_ind_loc[evol_ind_loc['ent'].isin(ent_vig['EENTIDAD'])]

evol_ind_loc_nan = evol_ind_loc.replace(0, np.nan)
# print(evol_ind_loc.loc[(evol_ind_loc['cod'] == 'RG1') & (evol_ind_loc['ent'] == '00011'), ['ent', 'cod', 'fecha', 'ind']])

import itertools

evol_ind_avg = evol_ind_loc_nan.drop(columns=['fecha']).groupby(['ent', 'cod'], as_index=False).mean()

# evol_ind_avg['ind'] = evol_ind_avg['ind'].fillna(0.0)

# evol_ind_avg['ind'] = evol_ind_avg['ind'].replace(np.nan, 0.0)

# evol_ind_avg = evol_ind_avg.reset_index()

# print(evol_ind_avg.isna().sum())

# CREO LA TABLA PIVOTE (COLUMNAS=INDICADORES, FILAS=ENTIDADES)

evol_ind_piv = evol_ind_avg.pivot(index='ent', columns='cod', values='ind').reset_index()
evol_ind_piv = evol_ind_piv.set_index('ent')

# print(evol_ind_piv)

gh = ent_vig.set_index('EENTIDAD')
evol_ind_piv = pd.concat([evol_ind_piv, gh['GH']], axis=1, join='inner')

evol_ind_piv = evol_ind_piv.fillna(0.0)
# print(evol_ind_piv.shape)
# print(list(set(evol_ind_piv['GH'])))
# print(evol_ind_piv.isna().sum())
# print(evol_ind_piv.iloc[0:1,])
# print(evol_ind_piv[evol_ind_piv['A21'].isna()])
# IMPORTO LIBRERIAS

from pandas.plotting import scatter_matrix
from pandas import DataFrame
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics


# LOF
# data visualzation
import matplotlib.pyplot as plt
import seaborn as sns
# outlier/anomaly detection
from sklearn.neighbors import LocalOutlierFactor

# model specification
model1 = LocalOutlierFactor(n_neighbors = 20, metric = "manhattan", contamination = 0.02)
# model fitting
y_pred = model1.fit_predict(evol_ind_piv)
# filter outlier index
outlier_index = np.where(y_pred == -1) # negative values are outliers and positives inliers
# filter outlier values
outlier_values = evol_ind_piv.iloc[outlier_index]


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
    predict['local outlier factor'] = -clf.decision_function(predict.iloc[:, :-1], novelty=False)
         # Separe los puntos del clúster de los puntos normales según el umbral
    outliers = predict[predict['local outlier factor'] > method].sort_values(by='local outlier factor')
    inliers = predict[predict['local outlier factor'] <= method].sort_values(by='local outlier factor')
    return outliers, inliers


# APPLYING LOF FUNCTION TO DROP OUTLIERS
# print(evol_ind_corr.head())
outlier_values, inlier_values = lof(evol_ind_piv)
# print(list_ent)
# print(list(outlier_values.index))
# print(list(inlier_values.index))
evol_ind_lof = evol_ind_piv[~evol_ind_piv.index.isin(outlier_values.index)]

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
plt.show()
