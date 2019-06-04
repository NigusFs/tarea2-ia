import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import os, psutil
import time

from sklearn.metrics import classification_report, confusion_matrix  
from mpl_toolkits.mplot3d import Axes3D

start_time = time.time()



def to_plot(X,name,color):

	fig = plt.figure(name,figsize=(8, 6))
	ax = Axes3D(fig)

	ax.set_xlabel('Inhibidores destruidos por el equipo_1')
	ax.set_ylabel('Barones eliminados por el equipo_1')
	ax.set_zlabel('Torres destruidas por el equipo_1')

	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color,s=60) 
	return ax

def asignador_colores(etiquetas,colores):
	aux=[]
	for elem in etiquetas:
		aux.append(colores[elem]) #le asigna un color a las etiquetas segun su valor
	return aux



#Dataset

dataframe = pd.read_csv('games.csv') #https://www.kaggle.com/datasnaek/league-of-legends

X = np.array(dataframe[["t1_inhibitorKills","t1_baronKills","t1_towerKills"]])
y = np.array(dataframe['winner']) #arreglo de 1 y 2, el cual indica el gandor de la partida.

plt.rcParams['figure.figsize'] = (24, 12)
plt.style.use('ggplot')
colours=[None,'blue','red'] #blue team 1, red team2
colores_Dataset=asignador_colores(y,colours)
to_plot(X,"Dataset",colores_Dataset)


###

#K-Means

k_means = KMeans(n_clusters=2, random_state=24).fit(X)
centroides = k_means.cluster_centers_
print("Ubicacion de los Centroides: \n",centroides,"\n")


y_Kmeans = k_means.predict(X)+1
C = k_means.cluster_centers_


colores_Kmeans=asignador_colores(y_Kmeans,colours)
ax = to_plot(X,"K-means",colores_Kmeans)
colours=['blue','red'] 
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colours, s=1000) #centroides



###

#Prediccion usando Regresion logica

regrL = LogisticRegression(C=1.0, solver='lbfgs', multi_class='ovr')
regrL.fit(X,y_Kmeans)

x1 = 8 #t1_inhibitorKills #si es cero la prob que gane el t1 deberia ser baja
y1 = 4 #t1_baronKills
z1 = 11 #t1_towerKills 

# Acotacion hay que elegir otros datos de la parttida, las variables del team 2 influyen mucho.

new_date =[x1,y1,z1]
new_predict = regrL.predict([new_date]) # ("t1_inhibitorKills","t1_baronKills","t1_towerKills")

print("Nuevos datos: ",new_date)
print("[RegLogic] Etiqueta del nuevo dato: ",int(new_predict),"\n")

## Pruebas para comparar los algoritmos 
print("Eficiencia:")
print ( "Tiempo de ejecucion: ", round(time.time() - start_time,3), "unidades de tiempo")
process = psutil.Process(os.getpid())
print( "Ram utilizada: ", round(process.memory_info().rss/1000000,2)," Mb\n")

print("Efectividad:")
print(classification_report(y,y_Kmeans))  

plt.show()
