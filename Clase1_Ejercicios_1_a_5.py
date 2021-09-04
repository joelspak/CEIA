# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 21:28:08 2021

@author: jspak
"""

import numpy as np

#se define la matriz A
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

# ej 1
def ej_1(a):
    i_0 = np.sum(a!=0, axis=1)
    p=1
    i_1 = np.sum(abs(a)**p, axis=1)**(1/p)
    p=2
    i_2 = np.sum(abs(a)**p, axis=1)**(1/p)
    i_3 = np.max(abs(a), axis=1)
    return i_0, i_1, i_2, i_3

i_0, i_1, i_2, i_3 = ej_1(a)

print("Resultados del ejercicio 1 con matriz A=")
print(str(a))
print("Norma 0: "+str(i_0))
print("Norma 1: "+str(i_1))
print("Norma 2: "+str(i_2))
print("Norma 3: "+str(i_3))


# ej 2
def ej_2(a):
    p=2
    i_2 = np.sum(abs(a)**p, axis=1)**(1/p)
    indices = np.argsort((-1)*i_2)
    nuevo_a = a[indices]
    return nuevo_a

print("Resultados del ejercicio 2 con matriz A=")
print(str(a))
print("Reordenamiento:")
print(ej_2(a))


#ej 3
def ej_3(id, idx, x):
    indices = np.where(id == x)
    return idx[indices]

def id2idx(id, idx, x):
    return ej_3(id, idx, x)

def idx2id(id, idx, x):
    return ej_3(idx, id, x)


idx = np.array([-1,4,5,-1,-1,-1,-1,-1,-1,-1,3,-1,1,-1,2,0])
id = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

x = np.array([15, 12, 14, 10, 1, 2, 1])
y = np.array([1,2,3,4,5,4])

print("Resultados del ejercicio 3: ")
for identificador in x:
    print("id2idx["+str(identificador)+"] -> "+str(id2idx(id, idx, identificador)))
for indice in y:
    print("idx2id["+str(indice)+"] -> "+str(idx2id(id, idx, indice)))
print("idx2id["+str(-1)+"] -> "+str(idx2id(id, idx, -1)))

#ej 4
def ej_4(truth, prediction):
    #le asigno dos puntos a la predicción, de forma tal que si sumo truth + prediction:
    #si truth=0,prediction=0 -> sum=0, es TN
    #si truth=1,prediction=0 -> sum=1, es FN
    #si truth=0,prediction=1 -> sum=2, es FP
    #si truth=1,prediction=1 -> sum=3, es TP
    result = truth + 2 * prediction
    TN = np.sum(result==0)
    FN = np.sum(result==1)
    FP = np.sum(result==2)
    TP = np.sum(result==3)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print("Resultados del ejercicio 4")
    print("Para la matriz de verdad "+str(truth)+" y la matriz de predicción "+str(prediction)+" tenemos:")
    print("Precisión = "+str(precision))
    print("Recall = "+str(recall))
    print("Accuracy = "+str(accuracy))
    
truth=np.array([1,1,0,1,1,1,0,0,0,1])
prediction=np.array([1,1,1,1,0,0,1,1,0,0])
ej_4(truth,prediction)


#ej 5
def ej_5(q_id, predicted_rank, truth_relevance):
    av_q_pr = 0
    ids = np.unique(q_id)
    for id in ids:
        indices=np.where(q_id==id)
        av_q_pr += np.sum(truth_relevance[indices])/(len(indices[0])*len(ids))
    
    return av_q_pr

q_id=np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
predicted_rank=np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])
truth_relevance=np.array([True, False, True, False, True, True, True, False, False, False, False, False, True, False, False, True])

av_q_pr=ej_5(q_id, predicted_rank, truth_relevance)
print("Resultados del ejercicio 5:")
print("Average query precision = "+str(av_q_pr))

