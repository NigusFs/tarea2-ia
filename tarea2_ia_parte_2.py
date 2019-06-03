#https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
#https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
#https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 

from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  

from sklearn.metrics import classification_report, confusion_matrix  
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


def to_plot(X,name,color):

	fig = plt.figure(name)
	ax = Axes3D(fig)

	ax.set_xlabel('Inhibidores destruidos por el equipo_1')
	ax.set_ylabel('Barones eliminados por el equipo_1')
	ax.set_zlabel('Torres destruidas por el equipo_1')

	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color,s=60) 
	return


dataframe = pd.read_csv('games.csv') #https://www.kaggle.com/datasnaek/league-of-legends
#print(dataframe.head())

X = np.array(dataframe[["t1_inhibitorKills","t1_baronKills","t1_towerKills"]])
y = np.array(dataframe['winner']) #arreglo de 1 y 2, el cual indica el gandor de la partida.

colours=['black','blue','red'] #blue team 1, red team2
aux=[]
for fila in y:
	aux.append(colours[fila]) #le asigna un color a la partida segun el ganador

to_plot(X,"Dataset",aux)


x1 = 0 #t1_inhibitorKills #si es cero la prob que gane el t1 deberia ser baja
y1 = 0 #t1_baronKills
z1 = 0 #t1_towerKills #pesa harto, hay q elegir otros datos de la parttida, las variables del team 2 influyen mucho
new_date =[x1,y1,z1]

knn =  KNeighborsClassifier(n_neighbors=4) 
knn.fit(X,y)
print(knn.predict([new_date])) #predicction work fine


#agregar ruido
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) #creo q esta parte agrupa, intenta graficar esto

colours=['black','blue','red']
aux=[]

for fila in y_train:
    aux.append(colours[fila])
 

to_plot(X_train,"Knn",aux) #i think this is not an agrupation of knn but who cares
plt.show()

#agregar ruido
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#scaler = StandardScaler()   #ni p**o idea pa q funciona esto
#scaler.fit(X_train)

#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)  

#classifier = KNeighborsClassifier(n_neighbors=4) 
#classifier.fit(X_train, y_train) 
#y_pred = classifier.predict(X_test) #prediccion, revisar que es X_test


#print(confusion_matrix(y_test, y_pred))  
#print(classification_report(y_test, y_pred))  
