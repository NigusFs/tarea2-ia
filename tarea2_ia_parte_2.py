import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  
from mpl_toolkits.mplot3d import Axes3D
import os, psutil
import time

from sklearn.metrics import classification_report, confusion_matrix

start_time = time.time()




def to_plot(X,name,color):

	fig = plt.figure(name,figsize=(8, 6))
	ax = Axes3D(fig)

	ax.set_xlabel('Inhibidores destruidos por el equipo_1')
	ax.set_ylabel('Barones eliminados por el equipo_1')
	ax.set_zlabel('Torres destruidas por el equipo_1')

	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color,s=60) 
	return

def asignador_colores(etiquetas,colores):
	aux=[]
	for elem in etiquetas:
		aux.append(colores[elem]) #le asigna un color a las etiquetas segun su valor
	return aux


## DataSet

dataframe = pd.read_csv('games.csv') #https://www.kaggle.com/datasnaek/league-of-legends

X = np.array(dataframe[["t1_inhibitorKills","t1_baronKills","t1_towerKills"]])
y = np.array(dataframe['winner']) #arreglo de 1 y 2, el cual indica el gandor de la partida.


plt.rcParams['figure.figsize'] = (24, 12)
plt.style.use('ggplot')
colours=[None,'blue','red'] #blue = team 1, red= team2
colores_Dataset=asignador_colores(y,colours)
to_plot(X,"Dataset",colores_Dataset)



# KNN

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=0) #Agrega ruido para evitar el overfiting

knn =  KNeighborsClassifier(n_neighbors=4) 
knn.fit(X_train,y_train)

y_knn = knn.predict(X_test) #Agrupa los datos 



colores_Knn=asignador_colores(y_knn,colours)
to_plot(X_test,"Knn",colores_Knn)


# Prediction
x1 = 8 #t1_inhibitorKills #si es cero la prob que gane el t1 deberia ser baja
y1 = 4 #t1_baronKills
z1 = 11 #t1_towerKills #pesa harto, hay q elegir otros datos de la parttida, las variables del team 2 influyen mucho
new_date =[x1,y1,z1]


print("Nuevos datos: ",new_date)
print("[Knn] La etiqueta del nuevo dato es :", knn.predict([new_date]),"\n") #predicction work fine 


## Pruebas para comparar los algoritmos 

print("Eficiencia:")
print ( "Tiempo de ejecucion: ", round(time.time() - start_time,3), "unidades de tiempo")
process = psutil.Process(os.getpid())
print( "Ram utilizada: ", round(process.memory_info().rss/1000000,2)," Mb\n")

print("Efectividad:") #depende de los datos de prueba y entrenamiento
print(classification_report(y_test,y_knn))  

plt.show()