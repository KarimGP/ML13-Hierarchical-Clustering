# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 08:49:37 2024

@author: KGP
"""

# Clustering jerárquico

# Cómo importar las librerías
import numpy as np # contiene las herrarmientas matemáticas para hacer los algoritmos de machine learning
import matplotlib.pyplot as plt #pyplot es la sublibrería enfocada a los gráficos, dibujos
import pandas as pd #librería para la carga de datos, manipular, etc

# Importar el dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values 


# Utilizar el dendograma para encontrar el número óptimo de clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

# Ajustar el clustering jeráriquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

# Visualización de los clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "red", label = "Cautos 1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "blue", label = "Estandar 2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "green", label = "Objetivo 3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = "cyan", label = "Descuidados 4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = "magenta", label = "Conservadores 5")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales ($)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()
