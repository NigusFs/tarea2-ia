import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

###parte 1

def to_plot(X,name,color):

	fig = plt.figure(name)
	ax = Axes3D(fig)

	ax.set_xlabel('Inhibidores destruidos por el equipo_1')
	ax.set_ylabel('Barones eliminados por el equipo_1')
	ax.set_zlabel('Torres destruidas por el equipo_1')

	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color,s=60) 
	return ax



#Dataset

dataframe = pd.read_csv('games.csv') #https://www.kaggle.com/datasnaek/league-of-legends
#print(dataframe.head())


X = np.array(dataframe[["t1_inhibitorKills","t1_baronKills","t1_towerKills"]])
y = np.array(dataframe['winner']) #arreglo de 1 y 2, el cual indica el gandor de la partida.


colours=['black','blue','red'] #blue team 1, red team2
aux=[]
for fila in y:
	aux.append(colours[fila]) #le asigna un color a la partida segun el ganador

#to_plot(X,"Dataset",aux)



###

#K-Means

k_means = KMeans(n_clusters=2).fit(X)
centroides = k_means.cluster_centers_
print(centroides)


labels = k_means.predict(X)
C = k_means.cluster_centers_
colours=['blue','red']
aux=[]

for fila in labels:
    aux.append(colours[fila])
 

#ax = to_plot(X,"K-means",aux)
#ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colours, s=1000) #centroides

###

#Regresion lineal

regrL = linear_model.LinearRegression()
regrL.fit(X,y)

#y_prediction = regrL.predict(X)

#print('Coefficients: \n', regrL.coef_)
# Error cuadrático medio
#print("Mean squared error: %.2f" % mean_squared_error(y, y_prediction))
# Evaluamos el puntaje de varianza (siendo 1.0 el mejor posible)
#print('Variance score: %.2f' % r2_score(y, y_prediction))

x1 = 0#t1_inhibitorKills #si es cero la prob que gane el t1 deberia ser baja
y1 =0 #t1_baronKills
z1 =0#t1_towerKills #pesa harto, hay q elegir otros datos de la parttida, las variables del team 2 influyen mucho
new_date =[x1,y1,z1]

new_predict = regrL.predict([new_date]) # ("t1_inhibitorKills","t1_baronKills","t1_towerKills")

print("Etiqueta del nuevo dato: ",int(new_predict))
#print(y)
#plt.show()

#######

#parte 2