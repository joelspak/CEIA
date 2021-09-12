# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:33:20 2021

@author: jspak
"""
import numpy as np
#ej 6

def ej_6(X,c):
    a,b = c.shape
    # hago un reshape de C con a vectores de 1xb
    c = c.reshape((a,1,b))
    #devuelvo la raíz cuadrada de la suma de los cuadrados de la diferencia entre las componentes.
    return np.sqrt(np.sum((X-c)**2, axis=-1))

# puntos
X=np.array([[1,2,3],[4,5,6],[7,8,9]])
### centroides
C=np.array([[1,0,0],[0,1,1]])

distancias = ej_6(X,C)

######## ej7 Obtener para cada fila en X, el índice de la fila en C con distancia euclídea más pequeña. Es decir, para cada fila en X, determinar a qué cluster pertenece en C. Hint: usar np.argmin.
def ej_7(X,c):
    return np.argmin(ej_6(X,c), axis=0)  

cluster = ej_7(X,C)

######### ej 8
def actualizar_centroides(X, n, clusters):
    centroides = []
    for i in range(n):
        a = (clusters != i)
        X_cluster = np.delete(X, a, 0)
        centroides.append(np.mean(X_cluster, axis=0))
    print(np.array(centroides))
    
    return np.array(centroides)
    
def ej_8(X,n):
    indices_aleatorios = np.random.choice(X.shape[0], size=n, replace=False)
    centroides = X[indices_aleatorios]
    print(centroides)
    for i in range(10):
        distancias = ej_6(X, centroides)
        clusterizacion = ej_7(X, centroides)
        centroides = actualizar_centroides(X, n, clusterizacion)
    return centroides, clusterizacion
    
n=2
X=np.array([[1,2,5],[1,5,6],[3,8,9],[4,5,7],[1,3,5],[2,8,6]])
ej_8(X,n)


